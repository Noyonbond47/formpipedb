# Forcing a new Vercel build on 2025-09-08
# Forcing a Vercel resync on 2025-09-08 at 11:03 PM

import os
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io, csv
import secrets
import re
from fastapi import FastAPI, Request, Header, HTTPException, status, Depends, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

# --- Google API Imports ---
from google.oauth2.credentials import Credentials as GoogleCredentials
from googleapiclient.discovery import build as build_google_service
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
from googleapiclient.errors import HttpError
from supabase import create_client, Client
from postgrest import APIError

# It's a good practice to load environment variables at the start
# In a real app, you'd use a library like python-dotenv for local development
# Vercel will inject these from your project settings
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = os.environ.get("SMTP_PORT")
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
CONTACT_FORM_RECIPIENT = os.environ.get("CONTACT_FORM_RECIPIENT")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

# Get the root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="FormPipeDB API - The Correct One")

# Mount the static files directory
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- API Models for the new Database -> Table structure ---
class DatabaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class DatabaseResponse(BaseModel):
    id: int
    created_at: str
    name: str
    description: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

class ForeignKeyDefinition(BaseModel):
    table_id: int
    column_name: str

class ColumnDefinition(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    # In the future, this can be expanded with more constraints
    type: str = Field(..., min_length=1) 
    is_primary_key: bool = False
    is_unique: bool = False
    is_not_null: bool = False
    foreign_key: Optional[ForeignKeyDefinition] = None

class TableCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    columns: List[ColumnDefinition]

class TableUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    columns: List[ColumnDefinition]

class TableResponse(BaseModel):
    id: int
    name: str
    columns: List[ColumnDefinition]

    model_config = ConfigDict(extra="ignore")

class RowResponse(BaseModel):
    id: int
    created_at: str
    table_id: int
    data: dict[str, Any]
    _meta: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")

class RowCreate(BaseModel):
    data: dict[str, Any]

class CsvPreviewRequest(BaseModel):
    csv_content: str

class CsvPreviewResponse(BaseModel):
    original_headers: List[str]
    sanitized_headers: List[str]
    inferred_types: List[str]

class CsvImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    csv_content: str
    # The user confirms the columns and types in the UI before sending
    columns: List[ColumnDefinition]

class PaginatedRowResponse(BaseModel):
    total: int
    data: List[RowResponse]

class QueryRequest(BaseModel):
    query: str

class CsvRowImportRequest(BaseModel):
    csv_content: str
    column_mapping: dict[str, str] # Maps table_column_name -> csv_header_name

class RowImportError(BaseModel):
    row_number: int
    data: dict[str, Any]
    error: str

class CsvRowImportResponse(BaseModel):
    inserted_count: int
    failed_count: int
    errors: List[RowImportError]

class SqlImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    script: str

class ContactForm(BaseModel):
    sender_name: str = Field(..., min_length=1)
    sender_email: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

# --- Webhook Models ---
class WebhookResponse(BaseModel):
    id: int
    created_at: str
    table_id: int
    webhook_token: str # This is a UUID but will be sent as a string
    status: str
    sample_payload: Optional[dict[str, Any]] = None
    field_mapping: Optional[dict[str, str]] = None

class WebhookUpdateRequest(BaseModel):
    # Only allow these specific status updates from the user
    status: Optional[str] = Field(None, pattern="^(active|disabled|listening)$")
    field_mapping: Optional[dict[str, str]] = None

class SqlTableCreateRequest(BaseModel):
    script: str

# --- Calendar Integration Models ---
class CalendarIntegrationFieldMapping(BaseModel):
    event_title_col: Optional[str] = None
    start_datetime_col: Optional[str] = None
    end_datetime_col: Optional[str] = None
    description_col: Optional[str] = None
    completed_status_col: Optional[str] = None # A boolean column

class CalendarIntegrationResponse(BaseModel):
    id: int
    table_id: int
    provider: str # e.g., 'google'
    account_email: str # The user's Google account email
    calendar_id: str # The specific calendar ID (e.g., 'primary')
    field_mapping: Optional[CalendarIntegrationFieldMapping] = None

    model_config = ConfigDict(extra="ignore")

class CalendarIntegrationCreate(BaseModel):
    provider: str = Field("google", pattern="^google$")
    account_email: str # For now, we'll mock this. In reality, it comes from OAuth.
    calendar_id: str
    field_mapping: CalendarIntegrationFieldMapping
    credentials: dict # This will hold the OAuth tokens from the frontend

class CalendarAutomationLogRow(BaseModel):
    row_id: int
    event_title: str
    event_start_time: str
    created_at: str # The timestamp of when the row was last updated/created

class GoogleOauthCodeRequest(BaseModel):
    code: str

# --- Reusable Dependencies ---
# This dependency handles getting the user's token, validating it, and providing the user object.
async def get_current_user_details(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authorization header missing or invalid"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # In a serverless environment like Vercel, creating a new client per request is a safe
        # and stateless pattern. The client is lightweight.
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        
        # Set the authorization for this client instance.
        # All subsequent requests with this client will be authenticated as the user.
        supabase.postgrest.auth(token)
        
        # Explicitly validate the JWT to ensure it's not expired or tampered with by fetching the user.
        # This call to Supabase Auth also returns the user's details.
        user_response = supabase.auth.get_user(token)
        user = user_response.user

        if not user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or user not found")
        
        # Return the authenticated client and user details
        return {"user": user, "client": supabase}
    except Exception as e:
        # This could be a PostgrestError or another exception
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid token: {str(e)}")

# --- API Endpoints ---
@app.get("/api/v1/databases", response_model=List[DatabaseResponse])
async def get_user_databases(auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches all top-level Databases for the logged-in user.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_databases").select("id, created_at, name, description").order("created_at", desc=True).execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/databases", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def create_user_database(db_data: DatabaseCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new Database for the logged-in user.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]
        new_db_data = {
            "user_id": user.id,
            "name": db_data.name,
            "description": db_data.description
        }
        response = supabase.table("user_databases").insert(new_db_data, returning="representation").execute()
        # The data is returned as a list, so we take the first element.
        return response.data[0]
    except APIError as e:
        # Check for PostgreSQL's unique violation error code
        if e.code == "23505":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A database with the name '{db_data.name}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create database: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/v1/databases/{database_id}", response_model=DatabaseResponse)
async def get_single_database(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches the details for a single database. RLS policy ensures the user owns it.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_databases").select("*").eq("id", database_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Database not found")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/databases/{database_id}/tables", response_model=List[TableResponse])
async def get_database_tables(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches all tables for a specific database.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").select("id, name, columns").eq("database_id", database_id).order("name").execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/databases/{database_id}/tables", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def create_database_table(database_id: int, table_data: TableCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table within a specific database.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]

        # Verify user has access to the parent database first
        db_check = supabase.table("user_databases").select("id").eq("id", database_id).maybe_single().execute()
        if not db_check.data:
            raise HTTPException(status_code=404, detail="Parent database not found or access denied")

        new_table_data = {
            "user_id": user.id,
            "database_id": database_id,
            "name": table_data.name,
            "columns": [col.dict() for col in table_data.columns]
        }
        insert_response = supabase.table("user_tables").insert(new_table_data, returning="representation").execute()
        # The data is returned as a list, so we take the first element.
        created_table = insert_response.data[0]

        # --- Automatically create a VIEW for this table ---
        # This makes the table immediately queryable in the SQL Runner.
        try:
            supabase.rpc('create_or_replace_view_for_table', {
                'p_table_id': created_table['id'],
                'p_table_name': created_table['name'],
                'p_columns': created_table['columns']
            }).execute()
        except Exception as view_error:
            # If view creation fails, we don't fail the whole request, but we should log it.
            print(f"Warning: Could not create view for table {created_table['id']}: {view_error}")

        return created_table
    except APIError as e:
        # Check for a unique constraint violation on the table name for that database
        if "user_tables_database_id_name_key" in str(e):
                 raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A table with the name '{table_data.name}' already exists in this database.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create table: {str(e)}")

@app.post("/api/v1/databases/{database_id}/create-table-from-sql", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def create_table_from_sql(database_id: int, sql_data: SqlTableCreateRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Parses a single CREATE TABLE SQL statement and creates the table within the specified database.
    This uses the same robust parser as the full SQL import.
    """
    supabase = auth_details["client"]
    script = sql_data.script.strip()

    if not script.upper().startswith("CREATE TABLE"):
        raise HTTPException(status_code=400, detail="Script must be a single CREATE TABLE statement.")

    # Use the same robust parser from the full import, but simplified for a single table
    create_match = re.search(r'CREATE TABLE\s+[`"]?(\w+)[`"]?\s*\((.+)\)', script, re.DOTALL | re.IGNORECASE)
    if not create_match:
        raise HTTPException(status_code=400, detail="Invalid CREATE TABLE syntax.")

    table_name, columns_str = create_match.groups()
    columns_defs = []
    table_level_fks = []

    # Split columns, being careful not to split inside parentheses like VARCHAR(255)
    for col_line in re.split(r',(?![^\(]*\))', columns_str):
        col_line = col_line.strip()
        if not col_line: continue

        fk_match = re.search(r'FOREIGN KEY\s*\(([`"]?\w+[`"]?)\)\s*REFERENCES\s*[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', col_line, re.IGNORECASE)
        if fk_match:
            source_col, ref_table_name, ref_col = fk_match.groups()
            table_level_fks.append({
                "source_col": source_col.strip('`"'),
                "ref_table_name": ref_table_name.strip('`"'),
                "ref_col": ref_col.strip('`"')
            })
            continue

        if col_line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT")):
            continue

        parts = col_line.split()
        if not parts: continue

        col_name = parts[0].strip('`"')
        type_and_constraints = " ".join(parts[1:])
        type_match = re.match(r'[\w\(\s,\)]+', type_and_constraints)
        col_type = type_match.group(0).strip() if type_match else parts[1]

        columns_defs.append(ColumnDefinition(
            name=col_name,
            type=col_type.lower(),
            is_primary_key="PRIMARY KEY" in type_and_constraints.upper(),
            is_unique="UNIQUE" in type_and_constraints.upper(),
            is_not_null="NOT NULL" in type_and_constraints.upper()
        ))

    # Create the table with basic columns
    table_create_payload = TableCreate(name=table_name, columns=columns_defs)
    created_table_dict = await create_database_table(database_id, table_create_payload, auth_details)
    created_table = TableResponse(**created_table_dict)

    # If there are foreign keys, update the table definition
    # This part is left for future enhancement if needed, as it requires resolving table names to IDs.
    # For now, the basic table creation is a huge improvement.

    return created_table

# Helper function for CSV import
def sanitize_header(header: str) -> str:
    # Lowercase, replace spaces/dashes with underscores, remove other invalid chars
    header = header.lower().strip()
    header = re.sub(r'[\s-]+', '_', header)
    header = re.sub(r'[^a-z0-9_]', '', header)
    # Ensure it's a valid identifier (doesn't start with a number)
    if header and header[0].isdigit():
        header = '_' + header
    return header

@app.post("/api/v1/import-csv/preview", response_model=CsvPreviewResponse)
async def preview_csv_import(preview_data: CsvPreviewRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Parses the start of a CSV to infer headers and column types for UI review.
    """
    try:
        csv_file = io.StringIO(preview_data.csv_content)
        reader = csv.reader(csv_file)
        
        original_headers = next(reader)
        sanitized_headers = [sanitize_header(h) for h in original_headers]
        if len(set(sanitized_headers)) != len(sanitized_headers):
            raise ValueError("CSV contains duplicate headers after sanitization.")

        sample_rows = [row for i, row in enumerate(reader) if i < 100]
        inferred_types = infer_column_types(sample_rows, len(sanitized_headers))

        return CsvPreviewResponse(original_headers=original_headers, sanitized_headers=sanitized_headers, inferred_types=inferred_types)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse CSV preview: {str(e)}")

def infer_column_types(rows: List[List[str]], num_cols: int) -> List[str]:
    """
    Infers column types by inspecting the first 100 rows of data.
    It attempts to find the most specific data type that fits all non-empty values in a column.
    The order of preference is: integer -> real -> boolean -> timestamp -> text.
    """
    from datetime import datetime

    # Define boolean string values (case-insensitive)
    BOOLEAN_TRUE_STRINGS = {'true', 't', 'yes', 'y', '1'}
    BOOLEAN_FALSE_STRINGS = {'false', 'f', 'no', 'n', '0'}

    def get_type(value: str):
        if not value:
            return None  # Ignore empty strings for type detection

        # Try integer
        try:
            int(value)
            return 'integer'
        except (ValueError, TypeError):
            pass

        # Try real (float)
        try:
            float(value)
            return 'real'
        except (ValueError, TypeError):
            pass

        # Try boolean
        if value.lower() in BOOLEAN_TRUE_STRINGS or value.lower() in BOOLEAN_FALSE_STRINGS:
            return 'boolean'

        # Try timestamp (common formats)
        for fmt in ('%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                datetime.strptime(value, fmt)
                return 'timestamp'
            except (ValueError, TypeError):
                pass

        return 'text'

    type_hierarchy = ['integer', 'real', 'boolean', 'timestamp', 'text']
    inferred_types = ['integer'] * num_cols

    for row in rows:
        if len(row) != num_cols: continue
        for i, cell in enumerate(row):
            current_type = inferred_types[i]
            if current_type == 'text' or not cell:
                continue

            cell_type = get_type(cell)
            if cell_type is None:
                continue

            # If the cell type is "less specific" than the current column type, upgrade the column type.
            if type_hierarchy.index(cell_type) > type_hierarchy.index(current_type):
                inferred_types[i] = cell_type

    return inferred_types

@app.post("/api/v1/databases/{database_id}/import-csv", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def import_table_from_csv(database_id: int, import_data: CsvImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table and populates it from a CSV file string.
    It infers column types and sanitizes headers.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    new_table_id = None

    # The user has already reviewed and confirmed the column types in the UI.
    # We receive the final column definitions directly in the payload.
    try:
        table_create_payload = TableCreate(name=import_data.name, columns=import_data.columns)
        created_table_dict = await create_database_table(database_id, table_create_payload, auth_details)
        created_table = TableResponse(**created_table_dict)
        new_table_id = created_table.id

        # Now, insert the data
        csv_file = io.StringIO(import_data.csv_content)
        
        # Use the sanitized headers from the confirmed columns payload
        sanitized_headers = [col.name for col in import_data.columns]
        column_types = {col.name: col.type for col in import_data.columns}

        # Rewind and read all data for insertion
        dict_reader = csv.DictReader(csv_file, fieldnames=sanitized_headers)
        next(dict_reader) # Skip header row

        rows_to_insert = []
        for row_dict in dict_reader:
            processed_row = {}
            for i, header in enumerate(sanitized_headers):
                val = row_dict.get(header)
                col_type = column_types.get(header, 'text')
                if val is not None and val != '':
                    if col_type == 'integer':
                        try: val = int(val)
                        except (ValueError, TypeError): pass
                    elif col_type == 'real':
                        try: val = float(val)
                        except (ValueError, TypeError): pass
                processed_row[header] = val
            rows_to_insert.append({"user_id": user.id, "table_id": new_table_id, "data": processed_row})

        if rows_to_insert:
            supabase.table("table_rows").insert(rows_to_insert).execute()

        return created_table
    except Exception as e:
        if new_table_id:
            supabase.table("user_tables").delete().eq("id", new_table_id).eq("user_id", user.id).execute()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to import CSV: {str(e)}")

@app.post("/api/v1/tables/{table_id}/import-rows-from-csv", response_model=CsvRowImportResponse)
async def import_rows_into_table(table_id: int, import_data: CsvRowImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Imports rows from a CSV file into an existing table based on a user-defined column mapping.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]

    try:
        # 1. Get the schema of the target table to know the expected data types
        table_schema_dict = await get_single_table(table_id, auth_details)
        table_schema = TableResponse(**table_schema_dict)
        table_column_types = {col.name: col.type for col in table_schema.columns}

        # 2. Parse the CSV
        csv_file = io.StringIO(import_data.csv_content)
        dict_reader = csv.DictReader(csv_file)
        
        rows_to_insert = []
        failed_rows = []

        for i, row_dict in enumerate(dict_reader, start=2): # Row 1 is header, so data starts at line 2
            processed_row_data = {}
            has_error = False
            error_reason = ""

            # 3. Map CSV columns to table columns and cast types
            for table_col, csv_header in import_data.column_mapping.items():
                if csv_header not in row_dict:
                    continue # Skip if the mapped CSV header doesn't exist in this row
                
                val = row_dict[csv_header]
                original_val = val
                target_type = table_column_types.get(table_col)

                if val is not None and val != '':
                    try:
                        if target_type == 'integer': val = int(val)
                        elif target_type == 'real': val = float(val)
                        elif target_type == 'boolean': val = val.lower() in {'true', 't', 'yes', 'y', '1'}
                        # Timestamps and text are kept as strings for insertion
                    except (ValueError, TypeError):
                        has_error = True
                        error_reason = f"Column '{table_col}': Invalid value '{original_val}' for type '{target_type}'."
                        break # Stop processing this row on first error
                
                processed_row_data[table_col] = val
            
            if has_error:
                failed_rows.append(RowImportError(row_number=i, data=row_dict, error=error_reason))
            elif processed_row_data:
                rows_to_insert.append({"user_id": user.id, "table_id": table_id, "data": processed_row_data})

        if rows_to_insert:
            supabase.table("table_rows").insert(rows_to_insert).execute()

        return CsvRowImportResponse(
            inserted_count=len(rows_to_insert),
            failed_count=len(failed_rows),
            errors=failed_rows
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to import rows: {str(e)}")

@app.put("/api/v1/tables/{table_id}", response_model=TableResponse)
async def update_database_table(table_id: int, table_data: TableUpdate, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates a table's structure (name and columns).
    """
    try:
        supabase = auth_details["client"]
        update_data = {
            "name": table_data.name,
            "columns": [col.dict() for col in table_data.columns]
        }
        response = supabase.table("user_tables").update(update_data, returning="representation").eq("id", table_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found or access denied.")
        return response.data[0]
    except APIError as e:
        # Handle case where the new table name conflicts with an existing one in the same database.
        if "user_tables_database_id_name_key" in str(e):
                 raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A table with the name '{table_data.name}' already exists in this database.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update table: {str(e)}")

@app.delete("/api/v1/databases/{database_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_database(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes a database and all its associated tables and rows.
    RLS and cascade delete handles the security and data integrity.
    """
    try:
        supabase = auth_details["client"]
        # The RLS policy ensures the user can only match their own database ID.
        # The 'returning="representation"' ensures data is returned to check if a row was actually deleted.
        response = supabase.table("user_databases").delete(returning="representation").eq("id", database_id).execute()
        
        if not response.data:
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Database not found or you do not have permission to delete it.")

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete database: {str(e)}")

@app.delete("/api/v1/tables/{table_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_database_table(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes a table and all its associated rows.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").delete(returning="representation").eq("id", table_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found or you do not have permission to delete it.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete table: {str(e)}")

@app.get("/api/v1/tables/{table_id}", response_model=TableResponse)
async def get_single_table(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches the details for a single table. RLS policy ensures the user owns it.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("user_tables").select("id, name, columns").eq("id", table_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Table not found")
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/tables/{table_id}/rows", response_model=PaginatedRowResponse)
async def get_table_rows(
    table_id: int, 
    auth_details: dict = Depends(get_current_user_details),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None)
):
    """
    Fetches data rows for a specific table with pagination and search.
    """
    supabase = auth_details["client"]
    try:
        # 1. Get the table schema to find the user-defined primary key column name
        table_schema_dict = await get_single_table(table_id, auth_details)
        # When calling an endpoint function directly, it returns a dict, not a Pydantic model.
        # We must convert it to a model to use attribute access.
        table_schema_obj = TableResponse(**table_schema_dict)
        pk_col_name = next((col.name for col in table_schema_obj.columns if col.is_primary_key), None)

        # 2. Build the query
        query = supabase.table("table_rows").select("*", count='exact').eq("table_id", table_id)

        if search:
            # Search across all non-pk columns by casting their JSONB value to text
            searchable_columns = [col.name for col in table_schema_obj.columns if not col.is_primary_key]
            if searchable_columns:
                or_filter = ",".join([f"data->>{col}.ilike.%{search}%" for col in searchable_columns])
                query = query.or_(or_filter)

        # RLS on table_rows ensures user can only access rows they own.
        response = query.order("id").range(offset, offset + limit - 1).execute()

        # 3. Process results to inject the user-visible PK
        processed_rows = []
        if pk_col_name:
            for i, row in enumerate(response.data):
                # Calculate the user-visible ID based on pagination
                user_visible_id = offset + i + 1
                
                # Inject it into the data blob
                if row.get("data") is not None:
                    row["data"][pk_col_name] = user_visible_id
                else:
                    row["data"] = {pk_col_name: user_visible_id}
                processed_rows.append(row)
        else:
            # Fallback if no PK is defined (shouldn't happen with current UI)
            processed_rows = response.data

        return {"total": response.count, "data": processed_rows}
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/tables/{table_id}/all-rows", response_model=List[RowResponse])
async def get_all_table_rows(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches ALL data rows for a specific table, bypassing pagination.
    Used for features like CSV export.
    """
    try:
        supabase = auth_details["client"]
        # 1. Get the table schema to find the user-defined primary key column name
        table_schema_dict = await get_single_table(table_id, auth_details)
        # When calling an endpoint function directly, it returns a dict, not a Pydantic model.
        # We must convert it to a model to use attribute access.
        table_schema_obj = TableResponse(**table_schema_dict)
        pk_col_name = next((col.name for col in table_schema_obj.columns if col.is_primary_key), None)

        # RLS on table_rows ensures user can only access rows they own.
        response = supabase.table("table_rows").select("*").eq("table_id", table_id).order("id").execute()

        # 2. Process results to inject the user-visible PK
        processed_rows = []
        if pk_col_name:
            for i, row in enumerate(response.data):
                user_visible_id = i + 1
                if row.get("data") is not None:
                    row["data"][pk_col_name] = user_visible_id
                else:
                    row["data"] = {pk_col_name: user_visible_id}
                processed_rows.append(row)
        else:
            processed_rows = response.data

        return processed_rows
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/tables/{table_id}/rows", response_model=RowResponse, status_code=status.HTTP_201_CREATED)
async def create_table_row(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new, empty row for a table.
    """
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]
        new_row_data = {
            "user_id": user.id,
            "table_id": table_id,
            "data": {} # Start with empty data
        }
        response = supabase.table("table_rows").insert(new_row_data, returning="representation").execute()
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create row: {str(e)}")

@app.put("/api/v1/rows/{row_id}", response_model=RowResponse)
async def update_table_row(row_id: int, row_data: RowCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates the data for a specific row.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("table_rows").update({"data": row_data.data}, returning="representation").eq("id", row_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or access denied.")
        
        # --- Trigger Calendar Automation ---
        # After a successful update, check if we need to create a calendar event.
        # We create an admin client here to run the logic with elevated privileges.
        if SUPABASE_SERVICE_ROLE_KEY:
            supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            await _create_or_update_calendar_event(row_id, supabase_admin)
        # --- End Trigger ---

        # Re-fetch the row to include any meta updates from the trigger
        final_response = supabase.table("table_rows").select("*").eq("id", row_id).single().execute()
        return final_response.data

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update row: {str(e)}")

@app.delete("/api/v1/rows/{row_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_table_row(row_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes a specific row.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("table_rows").delete(returning="representation").eq("id", row_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found or you do not have permission to delete it.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete row: {str(e)}")

# --- Webhook Management Endpoints ---

@app.post("/api/v1/tables/{table_id}/webhooks", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook_for_table(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new, unique webhook receiver for a specific table.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    try:
        # RLS on user_tables ensures the user owns the table they're creating a webhook for.
        # We don't need to check existence here because the INSERT RLS policy on public_webhooks does it.
        new_webhook_data = {
            "user_id": user.id,
            "table_id": table_id,
            # The webhook_token and status have default values in the DB ('listening').
        }
        response = supabase.table("public_webhooks").insert(new_webhook_data, returning="representation").execute()
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create webhook: {str(e)}")

@app.get("/api/v1/tables/{table_id}/webhooks", response_model=List[WebhookResponse])
async def get_webhooks_for_table(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Retrieves all webhooks configured for a specific table.
    """
    supabase = auth_details["client"]
    try:
        # RLS ensures user can only see webhooks for tables they own.
        response = supabase.table("public_webhooks").select("*").eq("table_id", table_id).order("created_at").execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.put("/api/v1/webhooks/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(webhook_id: int, update_data: WebhookUpdateRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates a webhook's status or field mapping.
    """
    supabase = auth_details["client"]
    try:
        # RLS on public_webhooks ensures the user can only update their own webhooks.
        payload = update_data.dict(exclude_unset=True)
        if not payload:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")
        
        # If user is resetting the webhook to re-map, clear out old config.
        if payload.get("status") == "listening":
            payload["sample_payload"] = None
            payload["field_mapping"] = None
        
        response = supabase.table("public_webhooks").update(payload, returning="representation").eq("id", webhook_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found or access denied.")
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update webhook: {str(e)}")

@app.delete("/api/v1/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(webhook_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes a specific webhook.
    """
    supabase = auth_details["client"]
    try:
        # RLS ensures the user can only delete their own webhooks.
        response = supabase.table("public_webhooks").delete(returning="representation").eq("id", webhook_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found or access denied.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete webhook: {str(e)}")

# --- Public Incoming Webhook Endpoint ---

@app.post("/api/v1/webhooks/incoming/{webhook_token}", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
async def handle_incoming_webhook(webhook_token: str, request: Request):
    """
    Public, unauthenticated endpoint to receive data from third-party services.
    It's hidden from the auto-generated API docs.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Webhook processing is not configured on the server.")

    # Use the service role key to bypass RLS and look up the webhook config.
    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    
    # 1. Find the webhook configuration
    try:
        webhook_config_res = supabase_admin.table("public_webhooks").select("*").eq("webhook_token", webhook_token).single().execute()
        webhook_config = webhook_config_res.data
    except APIError:
        # This happens if .single() finds no rows or more than one row.
        raise HTTPException(status_code=404, detail="Webhook not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook lookup failed: {str(e)}")

    # 2. Check webhook status and handle payload
    if webhook_config['status'] == 'disabled':
        return {"status": "ok", "message": "Webhook is disabled."} # Return 202 still, don't leak info.
    
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if webhook_config['status'] == 'listening':
        try:
            supabase_admin.table("public_webhooks").update({"sample_payload": payload}).eq("id", webhook_config['id']).execute()
            return {"status": "ok", "message": "Sample payload received."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save sample payload: {str(e)}")

    elif webhook_config['status'] == 'active':
        field_mapping = webhook_config.get('field_mapping')
        if not field_mapping:
            return {"status": "ok", "message": "Webhook is active but has no field mapping."}

        transformed_data = {table_col: payload.get(form_field) for table_col, form_field in field_mapping.items() if form_field in payload}
        
        if not transformed_data:
            return {"status": "ok", "message": "No mappable data found in payload."}

        try:
            insert_payload = { "user_id": webhook_config['user_id'], "table_id": webhook_config['table_id'], "data": transformed_data }
            insert_response = supabase_admin.table("table_rows").insert(insert_payload, returning="representation").execute()
            
            # --- Trigger Calendar Automation ---
            if insert_response.data:
                new_row_id = insert_response.data[0]['id']
                await _create_or_update_calendar_event(new_row_id, supabase_admin)
            # --- End Trigger ---

            return {"status": "ok", "message": "Data inserted."}
        except Exception as e:
            # Don't expose internal DB errors. Log this for debugging.
            print(f"Webhook insert error for token {webhook_token}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process data.")
    
    return {"status": "ok", "message": "Request received."}

# --- Calendar Integration Endpoints ---

@app.post("/api/v1/google/oauth2callback")
async def google_oauth2callback(code_request: GoogleOauthCodeRequest, request: Request, auth_details: dict = Depends(get_current_user_details)):
    """
    Exchanges a Google OAuth authorization code for credentials (access and refresh tokens).
    This is called by the frontend after the user grants consent.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google OAuth is not configured on the server.")

    try:
        # The redirect_uri must match exactly what's in your Google Cloud Console credentials
        # For this server-side exchange, it's not used for redirection, but for validation.
        # We'll use the request's origin as a flexible redirect_uri.
        redirect_uri = request.headers.get('origin')

        flow = Flow.from_client_config(
            client_config={
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=['https://www.googleapis.com/auth/calendar.events', 'openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            redirect_uri=redirect_uri)

        # This exchanges the code for credentials, which are stored on the flow object.
        flow.fetch_token(code=code_request.code)
        creds = flow.credentials

        # Add a more specific check to ensure the id_token was returned.
        # This fails if the 'openid' scope is missing.
        if not creds.id_token:
            raise HTTPException(status_code=400, detail="Google did not return an ID token. Ensure 'openid' scope is included in the request.")

        # Manually create a dictionary from the credentials object.
        # The 'Flow' object does not have a 'credentials_to_dict' method.
        # We also decode the id_token here on the server to securely get user info.
        try:
            id_info = id_token.verify_oauth2_token(creds.id_token, google_requests.Request(), GOOGLE_CLIENT_ID)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Google ID token: {e}")

        return {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'id_token_jwt': id_info, # The decoded token info, including email
            'scopes': creds.scopes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to exchange Google OAuth code: {str(e)}")

@app.get("/api/v1/tables/{table_id}/calendar-integration", response_model=Optional[CalendarIntegrationResponse])
async def get_calendar_integration(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Gets the active calendar integration for a table, if one exists.
    """
    # In a real app, you'd query the database:
    supabase = auth_details["client"]
    try:
        # Use .execute() which always returns a response object.
        # .maybe_single() can return None directly, causing an attribute error.
        response = supabase.table("calendar_integrations").select("*").eq("table_id", table_id).limit(1).execute()
        # If data is a list and it's not empty, return the first item. Otherwise, return None.
        # This correctly handles the "not found" case.
        return response.data[0] if response.data else None
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calendar integration: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching calendar settings: {str(e)}")

@app.post("/api/v1/tables/{table_id}/calendar-integration", response_model=CalendarIntegrationResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_calendar_integration(table_id: int, integration_data: CalendarIntegrationCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates or updates the calendar integration settings for a table.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    try:
        integration_payload = {
            "user_id": user.id,
            "table_id": table_id,
            "provider": integration_data.provider,
            "account_email": integration_data.account_email,
            "calendar_id": integration_data.calendar_id,
            "field_mapping": integration_data.field_mapping.dict(),
        }
        
        # Only include credentials in the payload if they are provided.
        # This allows updating the mapping without re-sending credentials.
        # IMPORTANT: In a production app, these credentials should be encrypted before saving.
        if integration_data.credentials:
            integration_payload["credentials"] = integration_data.credentials

        # Upsert on the table_id, which has a unique constraint. RLS ensures the user owns the table.
        response = supabase.table("calendar_integrations").upsert(integration_payload, on_conflict="table_id", returning="representation", ignore_duplicates=False).execute()
        
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=400, detail=f"Failed to save calendar integration: {str(e)}")

@app.delete("/api/v1/tables/{table_id}/calendar-integration", status_code=status.HTTP_204_NO_CONTENT)
async def delete_calendar_integration(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes the calendar integration for a table.
    """
    supabase = auth_details["client"]
    try:
        # RLS ensures the user can only delete their own integration.
        supabase.table("calendar_integrations").delete().eq("table_id", table_id).execute()
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete calendar integration: {str(e)}")

@app.get("/api/v1/tables/{table_id}/calendar-integration/logs", response_model=List[CalendarAutomationLogRow])
async def get_calendar_automation_logs(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Gets a log of the last 5 rows that had calendar events created for them.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        return [] # Return empty list if service key is not configured

    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    user_id = auth_details["user"].id

    try:
        # First, get the integration settings to know which column is the title/start time
        integration_res = supabase_admin.table("calendar_integrations").select("field_mapping").eq("table_id", table_id).eq("user_id", user_id).single().execute()
        if not integration_res.data or not integration_res.data.get("field_mapping"):
            return []

        mapping = integration_res.data["field_mapping"]
        title_col = mapping.get("event_title_col")
        start_col = mapping.get("start_datetime_col")

        if not title_col or not start_col:
            return []

        # Now, fetch the rows that have a calendar event ID in their meta field
        # Note the jsonb operators ->> to get field as text
        rows_res = supabase_admin.table("table_rows").select("id, updated_at, data").eq("table_id", table_id).not_.is_("(_meta->>calendar_event_id)", "null").order("updated_at", desc=True).limit(5).execute()
        
        # Format the response
        logs = [CalendarAutomationLogRow(row_id=row['id'], event_title=row['data'].get(title_col, "N/A"), event_start_time=row['data'].get(start_col, "N/A"), created_at=row['updated_at']) for row in rows_res.data]
        return logs

    except Exception as e:
        print(f"Error fetching calendar logs for table {table_id}: {e}")
        return [] # Return empty on error to not break the UI

async def _create_or_update_calendar_event(row_id: int, supabase_admin: Client):
    """Helper function to create a calendar event from a row if conditions are met."""
    # This function now contains the logic to call the actual Google Calendar API.
    # It assumes that the `google-api-python-client` library is installed and that
    # valid OAuth credentials have been stored in the `calendar_integrations` table.
    # The frontend is responsible for the OAuth flow to obtain these credentials.

    # 1. Fetch the row with its table_id and user_id.
    # We re-fetch here to ensure we have the absolute latest data, avoiding race conditions
    # where the calling function's data might be from just before the transaction committed.
    try:
        row_res = supabase_admin.table("table_rows").select("table_id, user_id, data, _meta").eq("id", row_id).single().execute()
        if not row_res.data: return
    except APIError:
        return # Row might have been deleted in the same transaction, which is fine.

    row = row_res.data
    table_id = row.get('table_id')
    user_id = row['user_id']
    row_data = row.get('data', {})
    row_meta = row.get('_meta', {}) or {}

    # 2. Fetch calendar integration for the table
    integration_res = supabase_admin.table("calendar_integrations").select("*").eq("table_id", table_id).eq("user_id", user_id).single().execute()
    if not integration_res.data: return

    integration = integration_res.data
    mapping = integration.get('field_mapping', {})
    credentials_dict = integration.get('credentials')

    # If credentials aren't stored or are still the mock ones, we can't proceed.
    if not credentials_dict or credentials_dict.get("mock"):
        return

    # 3. Check if required fields are present in the row data
    title_col = mapping.get('event_title_col')
    start_col = mapping.get('start_datetime_col')

    if not (title_col and start_col and row_data.get(title_col) and row_data.get(start_col)):
        return # Not enough data to create an event

    # 4. Check if an event has already been created to prevent duplicates
    if row_meta.get('calendar_event_id'):
        print(f"Row {row_id} already has a calendar event: {row_meta['calendar_event_id']}. Skipping.")
        return

    try:
        # 5. Build Google credentials and API service
        creds = GoogleCredentials.from_authorized_user_info(credentials_dict)
        service = build_google_service('calendar', 'v3', credentials=creds)

        # 6. Construct the event object for the Google API
        event_body = {
            "summary": row_data.get(title_col),
            "description": row_data.get(mapping.get('description_col')),
            "start": {"dateTime": row_data.get(start_col), "timeZone": "UTC"}, # Assuming UTC timezone
            "end": {"dateTime": row_data.get(mapping.get('end_datetime_col'), row_data.get(start_col)), "timeZone": "UTC"},
        }

        # 7. Call the Google Calendar API to insert the event
        print(f"Creating Google Calendar event for row {row_id}...")
        created_event = service.events().insert(calendarId=integration['calendar_id'], body=event_body).execute()
        
        real_event_id = created_event.get('id')
        if real_event_id:
            # 8. Update the row's meta data with the REAL event ID
            row_meta['calendar_event_id'] = real_event_id
            supabase_admin.table("table_rows").update({"_meta": row_meta}).eq("id", row_id).execute()
            print(f"Successfully created event {real_event_id} and updated row {row_id}.")

    except HttpError as error:
        print(f"An error occurred calling Google Calendar API for row {row_id}: {error}")
    except Exception as e:
        print(f"An unexpected error occurred during calendar event creation for row {row_id}: {e}")

@app.delete("/api/v1/users/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(auth_details: dict = Depends(get_current_user_details)):
    """
    Deletes the currently authenticated user's account from auth.users.
    This is an irreversible action. Requires the service_role key.
    Assumes that user data in public tables is set to cascade delete.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Server is not configured for user deletion."
        )

    user_id = auth_details["user"].id
    
    try:
        # An admin client is required to delete users.
        supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        supabase_admin.auth.admin.delete_user(user_id)
        
    except Exception as e:
        # This could be an APIError or another exception.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user account: {str(e)}"
        )


@app.get("/api/v1/databases/{database_id}/export-sql", response_class=PlainTextResponse)
async def export_database_as_sql(database_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Generates a full SQL script for a database, including CREATE TABLE and INSERT statements.
    """
    supabase = auth_details["client"]
    
    # Fetch database name
    db_res_dict = await get_single_database(database_id, auth_details)
    db_name = db_res_dict['name']

    # Fetch all tables for the database
    tables_dicts = await get_database_tables(database_id, auth_details)
    # Convert dicts to Pydantic models to safely use attribute access
    tables = [TableResponse(**t) for t in tables_dicts]
    # Create a map for efficient lookup of table names by ID
    table_id_to_name_map = {t.id: t.name for t in tables}
    
    sql_script = f"-- SQL Dump for database: {db_name}\n\n"

    for table in tables:
        # Generate CREATE TABLE statement
        sql_script += f"-- Structure for table: {table.name}\n"
        create_statement = f"CREATE TABLE \"{table.name}\" (\n"
        column_defs = []
        for col in table.columns:
            col_def = f"  \"{col.name}\" {col.type.upper()}"
            if col.is_primary_key: col_def += " PRIMARY KEY"
            if col.is_not_null: col_def += " NOT NULL"
            if col.is_unique: col_def += " UNIQUE"
            if col.foreign_key:
                referenced_table_name = table_id_to_name_map.get(col.foreign_key.table_id)
                if referenced_table_name:
                    col_def += f' REFERENCES "{referenced_table_name}" ("{col.foreign_key.column_name}")'
            column_defs.append(col_def)
        create_statement += ",\n".join(column_defs)
        create_statement += "\n);\n\n"
        sql_script += create_statement

        # Generate INSERT statements
        rows_dicts = await get_all_table_rows(table.id, auth_details)
        if rows_dicts:
            sql_script += f"-- Data for table: {table.name}\n"
            for row_dict in rows_dicts:
                # Skip rows that might not have any data in the JSONB field
                if not row_dict.get('data'):
                    continue

                # We only insert data from the 'data' blob. The user-visible PK is in here.
                # We need to find the actual PK column name to exclude it if it's auto-incrementing.
                pk_col_name = next((col.name for col in table.columns if col.is_primary_key), None)
                
                columns_to_insert = [f'"{k}"' for k in row_dict['data'].keys() if k != pk_col_name]
                values_to_insert = []
                for k, v in row_dict['data'].items():
                    if k == pk_col_name:
                        continue
                    if isinstance(v, str):
                        # Refactored for clarity: correctly escape single quotes for SQL
                        escaped_v = str(v).replace("'", "''")
                        values_to_insert.append(f"'{escaped_v}'")
                    elif v is None:
                        values_to_insert.append("NULL")
                    elif isinstance(v, bool):
                        values_to_insert.append("TRUE" if v else "FALSE")
                    else:
                        values_to_insert.append(str(v))

                if columns_to_insert:
                    sql_script += f"INSERT INTO \"{table.name}\" ({', '.join(columns_to_insert)}) VALUES ({', '.join(values_to_insert)});\n"
            sql_script += "\n"

    return PlainTextResponse(content=sql_script)

# This is the helper function from your initial snippet, fully implemented.
async def _parse_and_execute_insert(statement: str, created_tables_map: dict, db_id: int, supabase: Client, user: Any):
    # Handle multi-value INSERT statements
    header_match = re.search(r'INSERT INTO\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)\s*VALUES', statement, re.IGNORECASE)
    if not header_match: return
    
    table_name, columns_str = header_match.groups()
    if table_name not in created_tables_map: return
    
    table_id = created_tables_map[table_name]
    columns = [c.strip().strip('`"') for c in columns_str.split(',')]
    
    # Find all value tuples like (...), (...), (...) using a more robust, non-greedy regex
    values_part = statement[header_match.end():].strip()
    value_tuples_str = re.findall(r'\((.*?)\)', values_part)

    rows_to_insert = []
    for values_str in value_tuples_str:
        # Use the csv module to handle commas and quotes within values
        values_reader = csv.reader([values_str], skipinitialspace=True)
        values = next(values_reader)
        
        if len(columns) == len(values):
            # Attempt to convert numeric strings to actual numbers
            typed_values = []
            for v in values:
                try:
                    typed_values.append(int(v))
                except ValueError:
                    try:
                        typed_values.append(float(v))
                    except ValueError:
                        typed_values.append(v)
            
            rows_to_insert.append({"user_id": user.id, "table_id": table_id, "data": dict(zip(columns, typed_values))})

    if rows_to_insert:
        supabase.table("table_rows").insert(rows_to_insert).execute()

# This is the import endpoint from your initial snippet, fully implemented.
@app.post("/api/v1/databases/import-sql", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def import_database_from_sql(import_data: SqlImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new database and attempts to populate it from a user-provided SQL script.
    This has a very limited, best-effort SQL parser and is not guaranteed to work with all SQL dialects.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    new_db_id = None

    try:
        # 1. Create the parent database container
        db_response = await create_user_database(
            db_data=DatabaseCreate(name=import_data.name, description=import_data.description),
            auth_details=auth_details
        )
        new_db_id = db_response['id']

        # 2. Robust parsing and execution of the SQL script
        # First, remove all comments from the script
        script_no_comments = re.sub(r'--.*', '', import_data.script)
        statements = [s.strip() for s in script_no_comments.split(';') if s.strip()]
        
        # --- Multi-pass import process ---

        # Data structures to hold the parsed schema before creation
        parsed_tables = {} # { table_name: { "columns": [ColumnDefinition], "foreign_keys": [...] } }

        # Pass 1: Parse all CREATE TABLE statements and store their structure
        for statement in statements:
            if not statement.upper().startswith("CREATE TABLE"):
                continue

            create_match = re.search(r'CREATE TABLE\s+[`"]?(\w+)[`"]?\s*\((.+)\)', statement, re.DOTALL | re.IGNORECASE)
            if not create_match: continue
            
            table_name, columns_str = create_match.groups()
            columns_defs = []
            table_level_fks = []

            for col_line in columns_str.split(','):
                col_line = col_line.strip()
                if not col_line: continue

                # Check for table-level FOREIGN KEY constraint: FOREIGN KEY (col) REFERENCES other_table(other_col)
                fk_match = re.search(r'FOREIGN KEY\s*\(([`"]?\w+[`"]?)\)\s*REFERENCES\s*[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', col_line, re.IGNORECASE)
                if fk_match:
                    source_col, ref_table, ref_col = fk_match.groups()
                    table_level_fks.append({
                        "source_col": source_col.strip('`"'),
                        "ref_table": ref_table.strip('`"'),
                        "ref_col": ref_col.strip('`"')
                    })
                    continue

                # Skip other table-level constraints for now
                if col_line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT")):
                    continue

                # Assume it's a column definition
                parts = col_line.split()
                if not parts: continue

                col_name = parts[0].strip('`"')
                type_and_constraints = " ".join(parts[1:])
                type_match = re.match(r'[\w\(\s,\)]+', type_and_constraints)
                col_type = type_match.group(0).strip() if type_match else parts[1]

                columns_defs.append(ColumnDefinition(
                    name=col_name,
                    type=col_type.lower(),
                    is_primary_key="PRIMARY KEY" in type_and_constraints.upper(),
                    is_unique="UNIQUE" in type_and_constraints.upper(),
                    is_not_null="NOT NULL" in type_and_constraints.upper()
                ))
            
            if not columns_defs: continue
            parsed_tables[table_name] = {"columns": columns_defs, "foreign_keys": table_level_fks}

        # Pass 2: Create all tables (without FKs) to establish their IDs
        created_tables_map = {} # Maps table name to table ID
        for table_name, schema in parsed_tables.items():
            table_create_payload = TableCreate(name=table_name, columns=schema["columns"])
            created_table = await create_database_table(new_db_id, table_create_payload, auth_details)
            created_tables_map[table_name] = created_table['id']

        # Pass 3: Update tables with foreign key constraints
        all_created_tables_dicts = await get_database_tables(new_db_id, auth_details)
        all_created_tables_map = {t['name']: t for t in all_created_tables_dicts}

        for table_name, schema in parsed_tables.items():
            if not schema["foreign_keys"]: continue
            table_to_update_dict = all_created_tables_map.get(table_name)
            if not table_to_update_dict: continue
            
            table_to_update = TableResponse(**table_to_update_dict)
            made_changes = False
            for fk in schema["foreign_keys"]:
                target_column = next((c for c in table_to_update.columns if c.name == fk["source_col"]), None)
                referenced_table_id = created_tables_map.get(fk["ref_table"])
                if target_column and referenced_table_id:
                    target_column.foreign_key = ForeignKeyDefinition(table_id=referenced_table_id, column_name=fk["ref_col"])
                    made_changes = True

            if made_changes:
                update_payload = TableUpdate(name=table_to_update.name, columns=table_to_update.columns)
                await update_database_table(table_to_update.id, update_payload, auth_details)

        # Pass 4: Insert all data
        for statement in statements:
            if statement.upper().startswith("INSERT INTO"):
                await _parse_and_execute_insert(statement, created_tables_map, new_db_id, supabase, user)
        
        return db_response
    except Exception as e:
        # If any part of the process fails, roll back by deleting the created database.
        if new_db_id:
            await delete_user_database(new_db_id, auth_details)
        raise HTTPException(status_code=400, detail=f"Failed to import SQL script: {str(e)}. The new database has been rolled back.")

# This is the query execution logic from your initial snippet, fully implemented.
@app.post("/api/v1/databases/{database_id}/execute-query")
async def execute_custom_query(database_id: int, query_data: QueryRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Executes a read-only (SELECT) or data-modifying (INSERT, UPDATE, DELETE) SQL query
    from the user by calling a secure RPC function in the database. This supports
    complex queries like JOINs, provided that VIEWS have been created for the tables.
    """
    supabase = auth_details["client"]
    query = query_data.query.strip()

    # Remove single-line SQL comments. This is more robust than a simple regex
    # as it handles newlines correctly.
    uncommented_lines = [line for line in query.split('\n') if not line.strip().startswith('--')]
    query = "\n".join(uncommented_lines).strip()
    
    # 1. Clean the query by removing a single trailing semicolon.
    # A trailing semicolon is valid SQL syntax but causes an error when the query
    # is executed as a subquery inside the RPC function.
    if query.endswith(';'):
        query = query[:-1].strip()

    # 2. To make the query compatible with the simple parser in the `execute_query`
    # database function, we need to ensure there is no newline between the first
    # command (e.g., "SELECT") and the rest of the query.
    # This regex replaces the first block of whitespace with a single space, which is
    # safer than normalizing the whole string as it won't corrupt string literals.
    processed_query = re.sub(r'\s+', ' ', query, 1)

    try:
        # Call the 'execute_query' RPC function you created in the Supabase SQL Editor.
        # The function handles the secure execution of the query.
        response = supabase.rpc('execute_query', {'query_text': processed_query}).execute()
        # The RPC function returns a single JSON object, which could be an array of
        # results for SELECT, or a status object for DML.
        return response.data
    except Exception as e:
        # The error from the DB will be nested. We can try to extract a cleaner message.
        error_message = str(e)
        try:
            # PostgrestErrors have a 'message' attribute in their JSON representation
            import json
            error_details = json.loads(str(e))
            error_message = error_details.get('message', str(e))
        except:
            pass # Fallback to the raw error string
        
        raise HTTPException(status_code=400, detail=f"Query failed: {error_message}")


@app.post("/api/v1/contact")
async def handle_contact_form(form_data: ContactForm):
    """
    Handles submissions from the public contact form and sends an email.
    """
    # Check if the server is configured to send emails
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, CONTACT_FORM_RECIPIENT]):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Contact form is not configured on the server."
        )

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = CONTACT_FORM_RECIPIENT
    msg['Subject'] = f"New Contact Form Submission from {form_data.sender_name}"

    body = f"""
You have received a new message from your website's contact form:

Name: {form_data.sender_name}
Email: {form_data.sender_email}

Message:
{form_data.message}
    """
    msg.attach(MIMEText(body, 'plain'))

    # Send the email using smtplib
    try:
        server = smtplib.SMTP(SMTP_HOST, int(SMTP_PORT))
        server.starttls()  # Secure the connection
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, CONTACT_FORM_RECIPIENT, msg.as_string())
        server.quit()
        return {"message": "Message sent successfully!"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send message.")

# --- SEO / Static File Routes ---
@app.get("/robots.txt", response_class=FileResponse)
async def robots_txt():
    """Serves the robots.txt file from the public directory."""
    return FileResponse(path=BASE_DIR / "public" / "robots.txt", media_type="text/plain")

@app.get("/sitemap.xml", response_class=FileResponse)
async def sitemap_xml():
    """Serves the sitemap.xml file from the public directory."""
    return FileResponse(path=BASE_DIR / "public" / "sitemap.xml", media_type="application/xml")

# --- HTML Serving Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(
        "signup.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse(
        "forgot-password.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/update-password", response_class=HTMLResponse)
async def update_password_page(request: Request):
    return templates.TemplateResponse(
        "update-password.html", {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/email-verified", response_class=HTMLResponse)
async def email_verified_page(request: Request):
    # This page is shown after a user clicks the verification link in their email.
    return templates.TemplateResponse("email-verified.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    return templates.TemplateResponse(
        "app.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/app/database/{db_name}", response_class=HTMLResponse)
async def table_manager_page(request: Request, db_name: str):
    return templates.TemplateResponse(
        "table-manager.html",
        {
            "request": request,
            "db_name": db_name,
            "supabase_url": SUPABASE_URL,
            "supabase_anon_key": SUPABASE_ANON_KEY,
            "google_client_id": GOOGLE_CLIENT_ID,
        },
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})
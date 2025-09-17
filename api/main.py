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
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
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

# --- Calendar Models ---
class CalendarEventResponse(BaseModel):
    id: int
    title: str
    start_time: str
    end_time: Optional[str] = None
    description: Optional[str] = None
    is_completed: bool
    source_table_id: Optional[int] = None
    source_row_id: Optional[int] = None
    created_at: str

class CalendarEventUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    description: Optional[str] = None
    is_completed: Optional[bool] = None

class CalendarEventCreate(BaseModel):
    title: str
    start_time: str
    end_time: Optional[str] = None
    description: Optional[str] = None
    is_completed: bool = False
    source_table_id: int
    source_row_id: int

class RowToCalendarRequest(BaseModel):
    title_column: str
    start_time_column: str
    end_time_column: Optional[str] = None
    description_column: Optional[str] = None

class CalendarSyncConfig(BaseModel):
    is_enabled: bool
    column_mapping: RowToCalendarRequest

class CalendarSyncConfigResponse(BaseModel):
    id: int
    table_id: int
    is_enabled: bool
    column_mapping: Dict[str, Any]

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Script must be a single CREATE TABLE statement.")

    # A more robust regex to capture the table name and the columns block, even with complex content.
    create_match = re.search(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[`"]?(\w+)[`"]?\s*\((.*)\)\s*;?', script, re.DOTALL | re.IGNORECASE)
    if not create_match:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid CREATE TABLE syntax. Could not find table name and column definitions.")

    raw_table_name, columns_str = create_match.groups()
    # Sanitize the table name to lowercase to prevent case-sensitivity issues
    # when querying the view later.
    table_name = raw_table_name.lower()
    columns_defs = []
    table_level_fks = []

    # Split columns by comma, but be careful not to split inside parentheses (e.g., for VARCHAR(255) or DEFAULT functions)
    # This regex is more robust for this task.
    for col_line in re.split(r',(?![^()]*\))', columns_str):
        col_line = col_line.strip()
        if not col_line: continue

        # Clean up common redundant keywords from broken SQL dumps
        col_line = re.sub(r'\b(NOT NULL)\s+\1\b', r'\1', col_line, flags=re.IGNORECASE)
        col_line = re.sub(r'\b(UNIQUE)\s+\1\b', r'\1', col_line, flags=re.IGNORECASE)
        col_line = re.sub(r'\b(PRIMARY KEY)\s+\1\b', r'\1', col_line, flags=re.IGNORECASE)

        # Handle table-level foreign key definitions
        fk_match = re.search(r'FOREIGN KEY\s*\(([`"]?\w+[`"]?)\)\s*REFERENCES\s*[`"]?(\w+)[`"]?\s*\(([`"]?\w+[`"]?)\)', col_line, re.IGNORECASE)
        if fk_match:
            source_col, ref_table_name, ref_col = fk_match.groups()
            table_level_fks.append({
                "source_col": source_col.strip('`"\''),
                "ref_table_name": ref_table_name.strip('`"'),
                "ref_col": ref_col.strip('`"\'')
            })
            continue

        # Ignore other table-level constraints for now
        if col_line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CONSTRAINT", "CHECK")):
            continue

        # Extract column name, type, and constraints
        parts = col_line.split()
        if not parts: continue

        col_name = parts[0].strip('`"\'')
        type_and_constraints = " ".join(parts[1:]).strip()

        # --- FIX: Use the robust type extraction and normalization ---
        raw_col_type = _extract_sql_type(type_and_constraints)

        columns_defs.append(ColumnDefinition(
            name=col_name,
            type=normalize_sql_type(raw_col_type),
            is_primary_key="PRIMARY KEY" in type_and_constraints.upper(),
            is_unique="UNIQUE" in type_and_constraints.upper() and "PRIMARY KEY" not in type_and_constraints.upper(), # A PK is implicitly unique
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

def _extract_sql_type(col_def_str: str) -> str:
    """
    Robustly extracts the data type from a SQL column definition string.
    e.g., "INTEGER PRIMARY KEY NOT NULL" -> "INTEGER"
    e.g., "VARCHAR(255) UNIQUE" -> "VARCHAR(255)"
    """
    # List of known constraints to strip from the end
    constraints = [
        'PRIMARY KEY', 'NOT NULL', 'NULL', 'UNIQUE', 'DEFAULT .*', 'CHECK \(.*\)', 
        'REFERENCES .*', 'COLLATE .*'
    ]
    # The regex looks for the data type at the start, which might include parentheses.
    # It stops at the first known constraint keyword.
    match = re.match(r'^\s*([a-zA-Z_]+(?:\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)', col_def_str, re.IGNORECASE)
    return match.group(1) if match else 'TEXT'

def normalize_sql_type(sql_type: str) -> str:
    """Maps common SQL data types to the simplified types used by the app."""
    s_type = sql_type.lower()
    if 'int' in s_type:
        return 'integer'
    if 'char' in s_type or 'text' in s_type:
        return 'text'
    if 'real' in s_type or 'float' in s_type or 'double' in s_type or 'numeric' in s_type or 'decimal' in s_type:
        return 'real'
    if 'bool' in s_type:
        return 'boolean'
    if 'time' in s_type or 'date' in s_type:
        return 'timestamp'
    return 'text' # Default to text if no match

@app.post("/api/v1/databases/{database_id}/import-csv", response_model=TableResponse, status_code=status.HTTP_201_CREATED)
async def import_table_from_csv(database_id: int, import_data: CsvImportRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new table and populates it from a CSV file string.
    It infers column types and sanitizes headers.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]

    # --- FIX: Add a transaction for rollback on failure ---
    # This is a conceptual change. Supabase-py doesn't have explicit transactions,
    # but we will manually delete the table if a later step fails.

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
        
        updated_table = response.data[0]

        # --- FIX: Re-create the view to reflect the structure changes ---
        # This is the missing piece. Without this, the SQL Runner's view becomes outdated.
        try:
            supabase.rpc('create_or_replace_view_for_table', {
                'p_table_id': updated_table['id'],
                'p_table_name': updated_table['name'],
                'p_columns': updated_table['columns']
            }).execute()
        except Exception as view_error:
            print(f"Warning: Could not update view for table {updated_table['id']} after structure change: {view_error}")

        return updated_table
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
        # Ensure response.data is a list before iterating
        if response.data and isinstance(response.data, list):
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

async def _delete_item(supabase: Client, table_name: str, item_id: int, item_type_name: str):
    """Generic helper to delete an item from a table by ID, with RLS ensuring ownership."""
    try:
        response = supabase.table(table_name).delete(returning="representation").eq("id", item_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{item_type_name} not found or access denied.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete {item_type_name.lower()}: {str(e)}")


# --- Calendar Management Endpoints ---

@app.get("/api/v1/calendar/events", response_model=List[CalendarEventResponse])
async def get_calendar_events(
    start_date: str, # ISO 8601 format: YYYY-MM-DD
    end_date: str,   # ISO 8601 format: YYYY-MM-DD
    auth_details: dict = Depends(get_current_user_details)
):
    """
    Fetches all calendar events for the user within a given date range.
    """
    supabase = auth_details["client"]
    try:
        # This logic fetches events that OVERLAP with the requested date range.
        # An event overlaps if:
        # 1. It starts before the range ends (start_time <= end_date)
        # 2. It ends after the range starts (end_time >= start_date)
        # We also handle events with no end_time by treating their end as their start.
        response = supabase.rpc('get_calendar_events_in_range', {
            'p_start_date': start_date,
            'p_end_date': end_date
        }).execute()
        return response.data
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/calendar/events", response_model=CalendarEventResponse, status_code=status.HTTP_201_CREATED)
async def create_calendar_event(event_data: CalendarEventCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a new calendar event linked to an existing source row.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    try:
        new_event_data = event_data.dict()
        new_event_data["user_id"] = user.id

        # RLS on table_rows will ensure the user owns the source row, preventing unauthorized linking.
        # We don't need to explicitly check it here.

        response = supabase.table("calendar_events").insert(new_event_data, returning="representation").execute()
        if not response.data:
            raise APIError("Failed to create event.", code="500", message="Insert operation returned no data.")
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create event: {str(e)}")

@app.post("/api/v1/rows/{row_id}/send-to-calendar", response_model=CalendarEventResponse, status_code=status.HTTP_201_CREATED)
async def create_event_from_row(row_id: int, mapping: RowToCalendarRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates a calendar event by extracting data from an existing table row based on a provided column mapping.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    try:
        # 1. Fetch the source row data
        row_res = supabase.table("table_rows").select("table_id, data").eq("id", row_id).single().execute()
        if not row_res.data:
            raise HTTPException(status_code=404, detail="Source row not found.")
        
        row_data = row_res.data['data']
        table_id = row_res.data['table_id']

        # 2. Extract data based on the mapping
        title = row_data.get(mapping.title_column)
        start_time = row_data.get(mapping.start_time_column)

        if not title or not start_time:
            raise HTTPException(status_code=400, detail="Title and start time columns must exist in the row data and have values.")

        # 3. Construct the new event
        new_event_data = {
            "user_id": user.id,
            "title": str(title),
            "start_time": str(start_time),
            "end_time": row_data.get(mapping.end_time_column) if mapping.end_time_column else None,
            "description": str(row_data.get(mapping.description_column)) if mapping.description_column else None,
            "source_table_id": table_id,
            "source_row_id": row_id
        }

        # 4. Insert the new event
        response = supabase.table("calendar_events").insert(new_event_data, returning="representation").execute()
        return response.data[0]

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create event: {str(e)}")

@app.put("/api/v1/calendar/events/{event_id}", response_model=CalendarEventResponse)
async def update_calendar_event(event_id: int, update_data: CalendarEventUpdate, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates a calendar event. If the event is linked to a table row via auto-sync,
    it will also update the data in the source row, enabling bi-directional sync.
    """
    supabase = auth_details["client"]
    try:
        payload = update_data.dict(exclude_unset=True)
        if not payload:
            raise HTTPException(status_code=400, detail="No update data provided.")

        # --- Bi-directional Sync Logic ---
        # 1. Get the event's source info first to see if it's linked to a row.
        event_res = supabase.table("calendar_events").select("source_table_id, source_row_id").eq("id", event_id).single().execute()
        if event_res.data and event_res.data.get("source_table_id") and event_res.data.get("source_row_id"):
            source_table_id = event_res.data["source_table_id"]
            source_row_id = event_res.data["source_row_id"]

            # 2. Get the sync config for the source table to know the column mapping.
            sync_config_res = supabase.table("calendar_sync_configs").select("column_mapping").eq("table_id", source_table_id).maybe_single().execute()
            
            # Only write back to the source table if a sync config exists.
            if sync_config_res.data and sync_config_res.data.get("column_mapping"):
                mapping = sync_config_res.data["column_mapping"]
                
                # 3. Fetch the original row data.
                source_row_res = supabase.table("table_rows").select("data").eq("id", source_row_id).single().execute()
                if source_row_res.data:
                    updated_row_data = source_row_res.data['data'].copy()
                    
                    # 4. Map calendar changes back to the row data fields.
                    if 'title' in payload and mapping.get('title_column'): updated_row_data[mapping['title_column']] = payload['title']
                    if 'start_time' in payload and mapping.get('start_time_column'): updated_row_data[mapping['start_time_column']] = payload['start_time']
                    if 'end_time' in payload and mapping.get('end_time_column'): updated_row_data[mapping['end_time_column']] = payload['end_time']
                    if 'description' in payload and mapping.get('description_column'): updated_row_data[mapping['description_column']] = payload['description']
                    
                    # 5. Update the source row in the database.
                    supabase.table("table_rows").update({"data": updated_row_data}).eq("id", source_row_id).execute()

        # --- Original Logic: Update the calendar event itself ---
        response = supabase.table("calendar_events").update(payload, returning="representation").eq("id", event_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Event not found or access denied.")
        return response.data[0]
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not update event: {str(e)}")

@app.delete("/api/v1/calendar/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_calendar_event(event_id: int, auth_details: dict = Depends(get_current_user_details)):
    """Deletes a calendar event."""
    supabase = auth_details["client"]
    await _delete_item(supabase, "calendar_events", event_id, "Event")

@app.get("/api/v1/tables/{table_id}/calendar-sync", response_model=Optional[CalendarSyncConfigResponse])
async def get_calendar_sync_config(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Gets the automatic calendar sync configuration for a table.
    """
    supabase = auth_details["client"]
    try:
        response = supabase.table("calendar_sync_configs").select("id, table_id, is_enabled, column_mapping").eq("table_id", table_id).maybe_single().execute()
        # .maybe_single() can result in response.data being None if no row is found.
        # We also check if the response object itself is valid before accessing .data.
        # This handles cases where no config exists, preventing an error.
        if not response or response.data is None:
            return None
        return CalendarSyncConfigResponse(**response.data)
    except APIError as e:
        # This handles specific database errors (e.g., table not found)
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        # This is a general fallback for any other unexpected errors, like data validation issues.
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while fetching calendar config: {str(e)}")

@app.post("/api/v1/tables/{table_id}/calendar-sync", response_model=CalendarSyncConfigResponse)
async def create_or_update_calendar_sync_config(table_id: int, config_data: CalendarSyncConfig, auth_details: dict = Depends(get_current_user_details)):
    """
    Creates or updates the automatic calendar sync configuration for a table.
    If sync is enabled, this will also perform a backfill, creating calendar
    events for all existing rows in the table.
    """
    supabase = auth_details["client"]
    user = auth_details["user"]
    try:
        # Upsert ensures we create a new config or update an existing one for the table.
        upsert_payload = {
            "user_id": user.id,
            "table_id": table_id,
            "is_enabled": config_data.is_enabled,
            "column_mapping": config_data.column_mapping.dict(exclude_unset=True)
        }
        response = supabase.table("calendar_sync_configs").upsert(
            upsert_payload, 
            on_conflict="table_id", 
            returning="representation"
        ).execute()
        
        saved_config = response.data[0]

        # If sync was just enabled, trigger a backfill of existing rows to the calendar.
        if config_data.is_enabled:
            # 1. Fetch all rows from the source table.
            all_rows = await _get_all_table_rows_for_sync(table_id, auth_details)
            
            # 2. Prepare calendar events from these rows.
            events_to_upsert = []
            mapping = config_data.column_mapping
            
            for row in all_rows:
                row_data = row['data']
                title = row_data.get(mapping.title_column)
                start_time = row_data.get(mapping.start_time_column)

                # Skip rows that are missing essential data for a calendar event.
                if not title or not start_time:
                    continue

                events_to_upsert.append({
                    "user_id": user.id, "source_table_id": table_id, "source_row_id": row['id'], # Use the top-level 'id'
                    "title": str(title), "start_time": str(start_time),
                    "end_time": str(row_data.get(mapping.end_time_column)) if mapping.end_time_column and row_data.get(mapping.end_time_column) else None,
                    "description": str(row_data.get(mapping.description_column)) if mapping.description_column and row_data.get(mapping.description_column) else None,
                })

            # 3. Bulk upsert the events. This creates new ones and updates existing ones based on the source row.
            # This relies on a UNIQUE constraint on (user_id, source_table_id, source_row_id) in the calendar_events table.
            if events_to_upsert:
                supabase.table("calendar_events").upsert(
                    events_to_upsert, 
                    on_conflict="source_table_id, source_row_id"
                ).execute()

        return saved_config
    except APIError as e:
        raise HTTPException(status_code=400, detail=f"Could not save sync config: {str(e)}")

async def _get_all_table_rows_for_sync(table_id: int, auth_details: dict) -> List[Dict[str, Any]]:
    """
    Internal helper to fetch all rows for a table without any modification.
    This is used for backend processes like calendar backfill that need raw, unmodified data.
    It bypasses the response_model processing of the main API endpoint.
    """
    supabase = auth_details["client"]
    # RLS on table_rows ensures user can only access rows they own.
    # We fetch the raw data directly.
    response = supabase.table("table_rows").select("id, data").eq("table_id", table_id).order("id").execute()
    return response.data if response.data else []

@app.delete("/api/v1/tables/{table_id}/calendar-sync", status_code=status.HTTP_204_NO_CONTENT)
async def delete_calendar_sync_config(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """Deletes the calendar sync configuration for a table."""
    supabase = auth_details["client"]
    try:
        # RLS ensures the user can only delete their own sync configs.
        response = supabase.table("calendar_sync_configs").delete(returning="representation").eq("table_id", table_id).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sync config not found or access denied.")
    except APIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete sync config: {str(e)}")

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

            return {"status": "ok", "message": "Data inserted."}
        except Exception as e:
            # Don't expose internal DB errors. Log this for debugging.
            print(f"Webhook insert error for token {webhook_token}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process data.")
    
    return {"status": "ok", "message": "Request received."}

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
                    # The data from the DB is already a Python object.
                    # If it's a string that was part of the original import, it might have extra quotes.
                    # We need to handle the value as it is.
                    if isinstance(v, str):
                        # The value from JSONB is already a clean string. We just need to escape it for SQL.
                        escaped_v = v.replace("'", "''")
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

    raw_table_name, columns_str = header_match.groups()
    table_name = raw_table_name.lower() # Sanitize to lowercase
    if table_name not in created_tables_map:
        return # Silently skip if the table wasn't created in the first pass
    
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
                        # This is the key fix: strip leading/trailing quotes from string values
                        # that the CSV parser might leave.
                        clean_v = v.strip().strip("'\"")
                        typed_values.append(clean_v)
            
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
            
            raw_table_name, columns_str = create_match.groups()
            # Sanitize the table name to lowercase to prevent case-sensitivity issues.
            table_name = raw_table_name.lower()
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
                type_and_constraints = " ".join(parts[1:]).strip()
                # --- FIX: Use the robust type extraction ---
                col_type = _extract_sql_type(type_and_constraints)

                columns_defs.append(ColumnDefinition(
                    name=col_name,
                    type=normalize_sql_type(col_type),
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

@app.post("/api/v1/databases/{database_id}/execute-query")
async def execute_custom_query(database_id: int, query_data: QueryRequest, auth_details: dict = Depends(get_current_user_details)):
    """
    Executes a custom SQL query from the user via a secure RPC function.
    Supports data manipulation (SELECT, INSERT, UPDATE, DELETE) and CTEs.
    Blocks schema-modifying statements (CREATE, ALTER, DROP).
    """
    supabase = auth_details["client"]
    processed_query = query_data.query.strip()

    # 1. **Security Check**: Prevent schema modification.
    # Remove comments to prevent bypassing checks.
    query_no_comments = re.sub(r'--.*', '', processed_query)
    
    # Block keywords that modify schema or permissions.
    forbidden_keywords = r'\b(CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE)\b'
    if re.search(forbidden_keywords, query_no_comments, re.IGNORECASE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Schema-modifying statements (CREATE, ALTER, DROP, etc.) are not allowed in the SQL Runner. Please use the 'Structure' tab or 'Import' features."
        )

    # Ensure the query is a data manipulation or select statement.
    allowed_starts = ('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')
    if not query_no_comments.lstrip().upper().startswith(allowed_starts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only SELECT, WITH, INSERT, UPDATE, and DELETE queries are allowed in the SQL Runner."
        )

    # 2. **Execution**: The query is deemed safe for execution.
    # The `execute_dynamic_query` RPC function is designed to handle this.
    try:
        response = supabase.rpc('execute_dynamic_query', {
            'p_query_text': processed_query,
            'p_database_id': database_id
        }).execute()
        # The RPC function returns a properly formatted JSON object directly.
        # The supabase-python client wraps the single JSONB response in a list.
        # We need to extract the first element to return the object itself.
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            return response.data[0]
        return response.data # Fallback for other cases

    except APIError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Query failed: {e.message}")

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
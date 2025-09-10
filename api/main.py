# Forcing a new Vercel build on 2025-09-08
# Forcing a Vercel resync on 2025-09-08 at 11:03 PM

import os
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io, csv
import re
from fastapi import FastAPI, Request, Header, HTTPException, status, Depends, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Any
from pydantic import BaseModel, Field
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

class RowResponse(BaseModel):
    id: int
    created_at: str
    table_id: int
    data: dict[str, Any]

class RowCreate(BaseModel):
    data: dict[str, Any]

class PaginatedRowResponse(BaseModel):
    total: int
    data: List[RowResponse]

class QueryRequest(BaseModel):
    query: str

class SqlImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    script: str

class ContactForm(BaseModel):
    sender_name: str = Field(..., min_length=1)
    sender_email: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


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
        return response.data[0]
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

    # Remove SQL comments, which can interfere with parsing the query type.
    query = re.sub(r'--.*', '', query).strip()
    
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
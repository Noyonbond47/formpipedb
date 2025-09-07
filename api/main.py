import os
from pathlib import Path
from fastapi import FastAPI, Request, Header, HTTPException, status, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from supabase import create_client, Client

# It's a good practice to load environment variables at the start
# In a real app, you'd use a library like python-dotenv for local development
# Vercel will inject these from your project settings
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

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

# --- Reusable Dependencies ---
# This dependency handles getting the user's token, validating it, and providing the user object.
async def get_current_user_details(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing or invalid")
    
    token = authorization.split(" ")[1]
    
    try:
        # Create a new Supabase client for each request
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        
        # Set the authorization for this client instance.
        # All subsequent requests with this client will be authenticated as the user.
        supabase.postgrest.auth(token)
        
        # We still need to verify the token is valid and get user details
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or user not found")
        
        # Return the authenticated client and user details
        return {"user": user, "client": supabase}
    except Exception as e:
        # This could be a PostgrestError or other exception
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
    except Exception as e:
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
    except Exception as e:
        # Catch the specific error for duplicate names
        if "user_databases_user_id_name_key" in str(e):
             raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A database with the name '{db_data.name}' already exists.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create database: {str(e)}")

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
    except Exception as e:
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
    except Exception as e:
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
        response = supabase.table("user_tables").insert(new_table_data, returning="representation").execute()
        # The data is returned as a list, so we take the first element.
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"API v3 Error: Could not create table: {str(e)}")

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
        return response.data[0]
    except Exception as e:
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

    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/v1/tables/{table_id}/rows", response_model=List[RowResponse])
async def get_table_rows(table_id: int, auth_details: dict = Depends(get_current_user_details)):
    """
    Fetches all data rows for a specific table.
    """
    try:
        supabase = auth_details["client"]
        # RLS on table_rows ensures user can only access rows they own.
        response = supabase.table("table_rows").select("*").eq("table_id", table_id).order("id").execute()
        return response.data
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create row: {str(e)}")

@app.put("/api/v1/rows/{row_id}", response_model=RowResponse)
async def update_table_row(row_id: int, row_data: RowCreate, auth_details: dict = Depends(get_current_user_details)):
    """
    Updates the data for a specific row.
    """
    try:
        supabase = auth_details["client"]
        response = supabase.table("table_rows").update({"data": row_data.data}, returning="representation").eq("id", row_id).execute()
        return response.data[0]
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not delete row: {str(e)}")

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

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    return templates.TemplateResponse(
        "app.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/app/database/{database_id}", response_class=HTMLResponse)
async def database_detail_page(request: Request, database_id: int):
    return templates.TemplateResponse(
        "database_detail.html", 
        {"request": request, "database_id": database_id, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
    )

@app.get("/app/database/{database_id}/table/{table_id}", response_class=HTMLResponse)
async def table_detail_page(request: Request, database_id: int, table_id: int):
    return templates.TemplateResponse(
        "table_detail.html", 
        {
            "request": request, 
            "database_id": database_id,
            "table_id": table_id,
            "supabase_url": SUPABASE_URL, 
            "supabase_anon_key": SUPABASE_ANON_KEY
        }
    )

@app.get("/app/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    return templates.TemplateResponse(
        "profile.html", 
        {"request": request, "supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}
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
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

# --- API Models ---
# This defines the structure of a column within a form.
class ColumnDefinition(BaseModel):
    name: str
    type: str # e.g., 'text', 'number', 'date'

# This is the data we expect from the frontend when creating a form.
class FormCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    columns: List[ColumnDefinition]

# This defines the data structure for a form when we send it to the frontend.
class FormResponse(BaseModel):
    id: int
    created_at: str
    name: str
    description: Optional[str] = None
    columns: List[ColumnDefinition]

# --- Reusable Dependencies ---
# This dependency handles getting the user's token, validating it, and providing the user object.
async def get_current_user_details(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing or invalid")
    
    token = authorization.split(" ")[1]
    
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or user not found")
        
        return {"user": user, "token": token, "client": supabase}
    except Exception as e:
        # This could be a PostgrestError or other exception
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid token: {str(e)}")

# --- API Endpoints ---
# We'll prefix our data endpoints with /api/v1
@app.get("/api/v1/forms", response_model=List[FormResponse])
async def get_user_forms(auth_details: dict = Depends(get_current_user_details)):
    try:
        supabase = auth_details["client"]
        token = auth_details["token"]
        response = supabase.table("forms").select("id, created_at, name, description, columns").order("created_at", desc=True).auth(token).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/v1/forms", response_model=FormResponse, status_code=status.HTTP_201_CREATED)
async def create_user_form(form_data: FormCreate, auth_details: dict = Depends(get_current_user_details)):
    try:
        supabase = auth_details["client"]
        user = auth_details["user"]
        new_form_data = {
            "user_id": user.id,
            "name": form_data.name,
            "description": form_data.description,
            "columns": [col.dict() for col in form_data.columns]
        }
        response = supabase.table("forms").insert(new_form_data).select("*").single().execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not create form: {str(e)}")

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
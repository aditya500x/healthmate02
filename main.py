import sys
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Add current directory to path to ensure local modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
import uvicorn
import hashlib
import shutil

# Database ORM imports
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- External AI Module Integration ---
try:
    from analizerend.analizer import analyze_prescription_image
    ANALYZER_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"ANALYZEREND: AI features unavailable during startup: {e}")
    ANALYZER_AVAILABLE = False

# --- Database Configuration (PostgreSQL / Supabase) ---
# Fetches the URL from your .env file.
DATABASE_URL = os.getenv("DATABASE_URL")

# Handle potential 'postgres://' vs 'postgresql://' issues common with some cloud providers
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    print("WARNING: DATABASE_URL not found in .env. Falling back to local SQLite.")
    DATABASE_URL = "sqlite:///./healthmate.db"

# Create the SQLAlchemy engine
# psycopg2 is used as the driver for PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    uid = Column(Integer, unique=True, nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String)
    password = Column(String, nullable=False)
    role = Column(String, default="user")

# Create tables in Supabase (if they don't exist)
Base.metadata.create_all(bind=engine)

# --- Security & Hashing ---
def get_password_hash(password: str) -> str:
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks if the provided password matches the stored hash."""
    return get_password_hash(plain_password) == hashed_password

# --- Database Dependencies ---
def get_db():
    """Dependency to provide a database session to routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_next_uid(db: Session) -> int:
    """Generates the next sequential UID starting from 10000."""
    max_uid = db.query(func.max(User.uid)).scalar()
    if max_uid is None or max_uid < 10000:
        return 10000
    return max_uid + 1

# --- FastAPI Application Setup ---
app = FastAPI(title="HealthMate AI")

# CORS Middleware configuration
# In production, replace ["*"] with your specific Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize required local directories
for folder in ["static", "uploads", "templates"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_template_context(request: Request, user_name: str = "Anonymous", uid: int | None = None):
    """Generates the default context for Jinja2 templates."""
    error = request.query_params.get("error")
    return {"request": request, "user_name": user_name, "uid": uid, "error": error}

# --- AI Feature Endpoints ---

@app.post("/api/analyze-prescription")
async def analyze_prescription_endpoint(file: UploadFile = File(...)):
    """Processes uploaded prescription images via the AI module."""
    if not ANALYZER_AVAILABLE:
        return JSONResponse(
            {"message": "AI module unavailable on this server.", "medications": [], "interactions": [], "accuracy_score": 0.0},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    file_location = f"uploads/{file.filename}"
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run analysis
        result = analyze_prescription_image(file_location)
        return JSONResponse(result, status_code=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error during image analysis: {e}")
        return JSONResponse({"message": "Error processing image."}, status_code=500)
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_location):
            os.remove(file_location)

# --- Frontend View Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", get_template_context(request))

@app.get("/login", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", get_template_context(request))

@app.get("/signup", response_class=HTMLResponse)
async def read_signup(request: Request):
    return templates.TemplateResponse("signup.html", get_template_context(request))

@app.post("/login")
async def login_user(
    db: Session = Depends(get_db),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    """Authenticates users against the PostgreSQL database."""
    user = db.query(User).filter(User.email == email).first()
    
    if user and verify_password(password, user.password):
        if user.role == role:
            path = "/doctor_dashboard" if user.role == "doctor" else "/dashboard"
            return RedirectResponse(f"{path}?uid={user.uid}", status_code=status.HTTP_303_SEE_OTHER)
        return RedirectResponse(f"/login?error=Incorrect role selected for this account.", status_code=303)
    
    return RedirectResponse("/login?error=Invalid email or password.", status_code=303)

@app.post("/signup")
async def signup_user(request: Request, db: Session = Depends(get_db)):
    """Registers new users into the PostgreSQL database."""
    try:
        data = await request.json()
        if data.get('password') != data.get('confirm_password'):
            return JSONResponse({"message": "Passwords do not match."}, status_code=400)
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == data.get('email')).first()
        if existing_user:
            return JSONResponse({"message": "Email already registered."}, status_code=409)

        new_user = User(
            uid=get_next_uid(db),
            name=data.get('name'),
            email=data.get('email'),
            phone=data.get('phone'),
            password=get_password_hash(data.get('password')),
            role=data.get('role', 'user')
        )
        db.add(new_user)
        db.commit()
        
        target = '/doctor_dashboard' if new_user.role == 'doctor' else '/dashboard'
        return JSONResponse({"message": "Success", "redirect_url": f"{target}?uid={new_user.uid}"}, status_code=201)
    except Exception as e:
        print(f"Signup error: {e}")
        return JSONResponse({"message": "Internal server error during registration."}, status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request, uid: int | None = None, db: Session = Depends(get_db)):
    user_name = "Anonymous"
    if uid:
        user = db.query(User).filter(User.uid == uid).first()
        if user: 
            user_name = user.name
    return templates.TemplateResponse("dashboard.html", get_template_context(request, user_name, uid))

# --- Server Startup ---

if __name__ == "__main__":
    # Cloud Run injects the PORT environment variable
    server_port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=server_port, reload=True)

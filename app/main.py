from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ollama import Client as OllamaClient
from pydantic import BaseModel, EmailStr, Field
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_current_user, get_db, hash_password, verify_password
from app.config import settings
from app.db import engine
from app.graph import build_graph
from app.models import Base, Document, Report, User
from app.parsers import extract_text_for_file
from app.rag import index_document


app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

ollama_client = OllamaClient(host=settings.ollama_host)
qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
graph = build_graph(ollama_client=ollama_client, qdrant_client=qdrant_client)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".xlsx"}


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    Path(settings.storage_root).mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/analysis/{report_id}", response_class=HTMLResponse)
def analysis_page(request: Request, report_id: int):
    return templates.TemplateResponse("analysis.html", {"request": request, "report_id": report_id})


@app.post("/api/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(subject=str(user.id))
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "email": user.email}}


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token(subject=str(user.id))
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "email": user.email}}


@app.get("/api/me")
def me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}


@app.post("/api/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file extension")

    user_dir = Path(settings.storage_root) / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}{ext}"
    file_path = user_dir / filename

    file_bytes = file.file.read()
    file_path.write_bytes(file_bytes)

    document = Document(
        user_id=current_user.id,
        original_filename=file.filename or filename,
        storage_path=str(file_path),
        content_type=file.content_type or "application/octet-stream",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        extracted_text = extract_text_for_file(file_path)
        indexed_chunks = index_document(
            ollama_client=ollama_client,
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_model=settings.embedding_model,
            text=extracted_text,
            source=document.original_filename,
            user_id=current_user.id,
            document_id=document.id,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    except Exception as exc:
        db.delete(document)
        db.commit()
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File processing error: {exc}")

    return {
        "id": document.id,
        "filename": document.original_filename,
        "chunks_indexed": indexed_chunks,
    }


@app.get("/api/documents")
def list_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.scalars(
        select(Document).where(Document.user_id == current_user.id).order_by(Document.uploaded_at.desc())
    ).all()
    return [
        {
            "id": d.id,
            "filename": d.original_filename,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]


@app.post("/api/reports/generate")
def generate_report(
    payload: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = graph.invoke({"user_id": current_user.id, "question": payload.question})
    answer = result.get("answer", "BRAK_DANYCH")
    label = result.get("label", "NO_ANSWER")
    sources = [c.source for c in result.get("retrieved", [])]
    unique_sources = list(dict.fromkeys(sources))

    report = Report(
        user_id=current_user.id,
        question=payload.question,
        answer=answer,
        label=label,
        sources=unique_sources,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "id": report.id,
        "question": report.question,
        "answer": report.answer,
        "label": report.label,
        "sources": report.sources,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


@app.get("/api/reports/history")
def report_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reports = db.scalars(
        select(Report).where(Report.user_id == current_user.id).order_by(Report.created_at.desc())
    ).all()
    return [
        {
            "id": r.id,
            "question": r.question,
            "label": r.label,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]


@app.get("/api/reports/{report_id}")
def report_detail(report_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    report = db.scalar(select(Report).where(Report.id == report_id, Report.user_id == current_user.id))
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return {
        "id": report.id,
        "question": report.question,
        "answer": report.answer,
        "label": report.label,
        "sources": report.sources,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }

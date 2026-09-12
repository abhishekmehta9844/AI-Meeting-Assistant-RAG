from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import bot, meetings, chat, auth
from app.models.meeting import Meeting
from app.models.user import User
from app.core.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Meeting Assistant API",
    description="Backend API for the RAG-based Meeting Assistant",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(meetings.router, prefix="/api/meetings", tags=["Meetings"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(bot.router, prefix="/api/bot", tags=["Bot"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])

@app.get("/")
def root():
    return {"message": "Welcome to the AI Meeting Assistant API"}


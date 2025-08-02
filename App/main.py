from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from App.routers import merge
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting HackRX RAG API...")
    yield
    # Shutdown
    logger.info("Shutting down HackRX RAG API...")

app = FastAPI(
    title="HackRX RAG API",
    description="Enhanced Document Q&A System with Adaptive RAG",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon - in production, specify actual origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# Include routers
app.include_router(merge.router)

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "HackRX RAG API is operational!",
        "version": "2.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": "2025-08-02T14:00:00Z",
        "services": {
            "rag_system": "operational",
            "document_processor": "operational",
            "vector_store": "operational"
        }
    }

# Enhanced OpenAPI schema with proper security
security = HTTPBearer()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="HackRX RAG API",
        version="2.0.0",
        description="Enhanced Document Q&A System with Adaptive RAG for variable-sized PDFs",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your bearer token"
        }
    }
    
    # Apply security to all protected endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if path != "/" and path != "/health":  # Skip public endpoints
                if "security" not in openapi_schema["paths"][path][method]:
                    openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="An unexpected error occurred. Please try again later."
    )

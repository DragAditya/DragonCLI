from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

from app.core.config import settings
from app.core.database import engine, create_tables
from app.core.security import get_current_user
from app.api.routes import auth, sandboxes, filesystem, ai, websocket, admin
from app.services.container_service import ContainerService
from app.services.ai_service import AIOrchestrator
from app.monitoring.metrics import MetricsCollector
from app.core.exceptions import (
    CustomException,
    custom_exception_handler,
    validation_exception_handler,
    http_exception_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
container_service = ContainerService()
ai_orchestrator = AIOrchestrator()
metrics_collector = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the FastAPI application."""
    # Startup
    logger.info("Starting Terminal++ Backend...")
    
    # Create database tables
    await create_tables()
    
    # Initialize services
    await container_service.initialize()
    await ai_orchestrator.initialize()
    
    # Start background tasks
    await container_service.start_monitoring()
    await ai_orchestrator.start_background_tasks()
    
    logger.info("Backend startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Terminal++ Backend...")
    
    # Cleanup services
    await container_service.cleanup()
    await ai_orchestrator.cleanup()
    
    logger.info("Backend shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="Terminal++ API",
    description="Cloud-native AI-powered development terminal",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Security middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(CustomException, custom_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(422, validation_exception_handler)

# Include API routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(sandboxes.router, prefix="/api/sandboxes", tags=["Sandboxes"])
app.include_router(filesystem.router, prefix="/api/fs", tags=["Filesystem"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Terminal++ API",
        "version": "1.0.0",
        "description": "Cloud-native AI-powered development terminal",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        
        # Check Redis connection
        # TODO: Add Redis health check
        
        # Check container service
        container_status = await container_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "services": {
                "database": "healthy",
                "redis": "healthy",
                "container_service": "healthy" if container_status else "unhealthy",
                "ai_service": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/api/stats")
async def get_stats(current_user = Depends(get_current_user)):
    """Get system statistics."""
    try:
        stats = await container_service.get_system_stats()
        return {
            "containers": stats,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        websocket = self.active_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for websocket in self.active_connections.values():
            await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/terminal/{session_id}")
async def websocket_terminal(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for terminal communication."""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process terminal input/output
            if data["type"] == "input":
                # Send input to container
                await container_service.send_input(session_id, data["data"])
            elif data["type"] == "resize":
                # Resize terminal
                await container_service.resize_terminal(
                    session_id, 
                    data["cols"], 
                    data["rows"]
                )
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        await container_service.cleanup_session(session_id)

@app.websocket("/ws/ai/{session_id}")
async def websocket_ai(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for AI communication."""
    await manager.connect(websocket, f"ai_{session_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process AI request
            response = await ai_orchestrator.process_request(
                session_id=session_id,
                message=data["message"],
                context=data.get("context", {}),
                mode=data.get("mode", "review")
            )
            
            await websocket.send_json({
                "type": "ai_response",
                "data": response
            })
            
    except WebSocketDisconnect:
        manager.disconnect(f"ai_{session_id}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )
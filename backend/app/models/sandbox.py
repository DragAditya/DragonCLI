from sqlalchemy import String, Integer, DateTime, Enum, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import enum

from app.core.database import Base

class SandboxStatus(str, enum.Enum):
    CREATING = "creating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATING = "terminating"

class SandboxTemplate(str, enum.Enum):
    UBUNTU = "ubuntu"
    PYTHON = "python-dev"
    NODEJS = "nodejs-dev"
    GOLANG = "golang-dev"
    RUST = "rust-dev"
    DATA_SCIENCE = "data-science"
    WEB_DEV = "web-dev"
    DOCKER = "docker"
    CUSTOM = "custom"

class Sandbox(Base):
    __tablename__ = "sandboxes"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    template: Mapped[SandboxTemplate] = mapped_column(
        Enum(SandboxTemplate), 
        default=SandboxTemplate.UBUNTU
    )
    
    # Container info
    container_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    container_name: Mapped[Optional[str]] = mapped_column(String(255))
    image: Mapped[str] = mapped_column(String(255), default="ubuntu:22.04")
    
    # Status
    status: Mapped[SandboxStatus] = mapped_column(
        Enum(SandboxStatus), 
        default=SandboxStatus.STOPPED,
        index=True
    )
    status_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Resource configuration
    cpu_limit: Mapped[int] = mapped_column(Integer, default=1000)  # millicores
    memory_limit: Mapped[int] = mapped_column(Integer, default=512)  # MB
    disk_limit: Mapped[int] = mapped_column(Integer, default=1024)  # MB
    network_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Environment
    environment_vars: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, default=dict)
    startup_script: Mapped[Optional[str]] = mapped_column(Text)
    working_directory: Mapped[str] = mapped_column(String(500), default="/workspace")
    
    # Access control
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    password_protected: Mapped[bool] = mapped_column(Boolean, default=False)
    access_password: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Configuration
    ports: Mapped[Optional[List[int]]] = mapped_column(JSON, default=list)
    volumes: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, default=dict)
    packages: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    
    # Settings
    auto_save: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_sleep_minutes: Mapped[int] = mapped_column(Integer, default=30)
    auto_delete_days: Mapped[int] = mapped_column(Integer, default=30)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Statistics
    total_runtime_minutes: Mapped[int] = mapped_column(Integer, default=0)
    cpu_usage_history: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, default=list)
    memory_usage_history: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, default=list)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sandboxes")
    sessions: Mapped[List["Session"]] = relationship(
        "Session",
        back_populates="sandbox",
        cascade="all, delete-orphan"
    )
    file_versions: Mapped[List["FileVersion"]] = relationship(
        "FileVersion",
        back_populates="sandbox",
        cascade="all, delete-orphan"
    )
    snapshots: Mapped[List["Snapshot"]] = relationship(
        "Snapshot",
        back_populates="sandbox",
        cascade="all, delete-orphan"
    )
    collaborations: Mapped[List["Collaboration"]] = relationship(
        "Collaboration",
        back_populates="sandbox",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Sandbox(id={self.id}, name={self.name}, status={self.status.value})>"
    
    def to_dict(self):
        """Convert sandbox to dictionary for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "template": self.template.value,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "image": self.image,
            "status": self.status.value,
            "status_message": self.status_message,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "disk_limit": self.disk_limit,
            "network_enabled": self.network_enabled,
            "environment_vars": self.environment_vars or {},
            "startup_script": self.startup_script,
            "working_directory": self.working_directory,
            "is_public": self.is_public,
            "password_protected": self.password_protected,
            "ports": self.ports or [],
            "volumes": self.volumes or {},
            "packages": self.packages or [],
            "auto_save": self.auto_save,
            "auto_sleep_minutes": self.auto_sleep_minutes,
            "auto_delete_days": self.auto_delete_days,
            "total_runtime_minutes": self.total_runtime_minutes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None
        }
    
    def to_public_dict(self):
        """Convert sandbox to public dictionary (for sharing)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template": self.template.value,
            "image": self.image,
            "status": self.status.value,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }
    
    @property
    def is_running(self) -> bool:
        """Check if sandbox is currently running."""
        return self.status == SandboxStatus.RUNNING
    
    @property
    def is_accessible(self) -> bool:
        """Check if sandbox can be accessed."""
        return self.status in [SandboxStatus.RUNNING, SandboxStatus.PAUSED]
    
    @property
    def uptime_minutes(self) -> int:
        """Calculate current uptime in minutes."""
        if not self.started_at or not self.is_running:
            return 0
        return int((datetime.utcnow() - self.started_at).total_seconds() / 60)
    
    @property
    def idle_minutes(self) -> int:
        """Calculate idle time in minutes."""
        return int((datetime.utcnow() - self.last_accessed).total_seconds() / 60)
    
    @property
    def should_auto_sleep(self) -> bool:
        """Check if sandbox should be auto-slept."""
        return (
            self.is_running and 
            self.auto_sleep_minutes > 0 and 
            self.idle_minutes >= self.auto_sleep_minutes
        )
    
    @property
    def should_auto_delete(self) -> bool:
        """Check if sandbox should be auto-deleted."""
        return (
            self.auto_delete_days > 0 and 
            self.idle_minutes >= (self.auto_delete_days * 24 * 60)
        )
    
    def update_access_time(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
    
    def start(self):
        """Mark sandbox as started."""
        self.status = SandboxStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.stopped_at = None
        self.update_access_time()
    
    def stop(self):
        """Mark sandbox as stopped."""
        self.status = SandboxStatus.STOPPED
        self.stopped_at = datetime.utcnow()
        
        # Update total runtime
        if self.started_at:
            runtime = int((self.stopped_at - self.started_at).total_seconds() / 60)
            self.total_runtime_minutes += runtime
    
    def pause(self):
        """Mark sandbox as paused."""
        self.status = SandboxStatus.PAUSED
        
        # Update total runtime
        if self.started_at:
            runtime = int((datetime.utcnow() - self.started_at).total_seconds() / 60)
            self.total_runtime_minutes += runtime
    
    def error(self, message: str):
        """Mark sandbox as in error state."""
        self.status = SandboxStatus.ERROR
        self.status_message = message
        self.stopped_at = datetime.utcnow()
    
    def add_usage_data(self, cpu_percent: float, memory_mb: float):
        """Add resource usage data point."""
        timestamp = datetime.utcnow().isoformat()
        
        # CPU usage
        if not self.cpu_usage_history:
            self.cpu_usage_history = []
        self.cpu_usage_history.append({
            "timestamp": timestamp,
            "value": cpu_percent
        })
        
        # Memory usage
        if not self.memory_usage_history:
            self.memory_usage_history = []
        self.memory_usage_history.append({
            "timestamp": timestamp,
            "value": memory_mb
        })
        
        # Keep only last 100 data points
        if len(self.cpu_usage_history) > 100:
            self.cpu_usage_history = self.cpu_usage_history[-100:]
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        cpu_values = [point["value"] for point in (self.cpu_usage_history or [])]
        memory_values = [point["value"] for point in (self.memory_usage_history or [])]
        
        return {
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "history": self.cpu_usage_history or []
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "history": self.memory_usage_history or []
            },
            "uptime_minutes": self.uptime_minutes,
            "total_runtime_minutes": self.total_runtime_minutes,
            "idle_minutes": self.idle_minutes
        }
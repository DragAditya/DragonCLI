from sqlalchemy import String, Boolean, DateTime, Enum, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List
import enum

from app.core.database import Base

class UserTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class User(Base):
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Profile
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(100))
    website: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Subscription
    tier: Mapped[UserTier] = mapped_column(Enum(UserTier), default=UserTier.FREE)
    subscription_id: Mapped[Optional[str]] = mapped_column(String(100))
    subscription_expires: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Usage limits
    max_sandboxes: Mapped[int] = mapped_column(default=5)
    max_storage_mb: Mapped[int] = mapped_column(default=1024)  # 1GB
    max_cpu_time_minutes: Mapped[int] = mapped_column(default=600)  # 10 hours
    
    # Settings
    preferences: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Security
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    two_factor_secret: Mapped[Optional[str]] = mapped_column(String(32))
    backup_codes: Mapped[Optional[dict]] = mapped_column(JSON)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    login_count: Mapped[int] = mapped_column(default=0)
    
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
    
    # Relationships
    sandboxes: Mapped[List["Sandbox"]] = relationship(
        "Sandbox", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    sessions: Mapped[List["Session"]] = relationship(
        "Session",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    collaborations: Mapped[List["Collaboration"]] = relationship(
        "Collaboration",
        foreign_keys="[Collaboration.user_id]",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    invited_collaborations: Mapped[List["Collaboration"]] = relationship(
        "Collaboration",
        foreign_keys="[Collaboration.invited_by]",
        back_populates="inviter"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    def to_dict(self):
        """Convert user to dictionary for API responses."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "tier": self.tier.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "two_factor_enabled": self.two_factor_enabled,
            "preferences": self.preferences or {},
            "notifications_enabled": self.notifications_enabled,
            "max_sandboxes": self.max_sandboxes,
            "max_storage_mb": self.max_storage_mb,
            "max_cpu_time_minutes": self.max_cpu_time_minutes,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "login_count": self.login_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_public_dict(self):
        """Convert user to public dictionary (without sensitive data)."""
        return {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat()
        }
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium subscription."""
        return self.tier in [UserTier.PRO, UserTier.ENTERPRISE]
    
    @property
    def has_valid_subscription(self) -> bool:
        """Check if user has valid subscription."""
        if self.tier == UserTier.FREE:
            return True
        if not self.subscription_expires:
            return False
        return self.subscription_expires > datetime.utcnow()
    
    def can_create_sandbox(self, current_sandbox_count: int) -> bool:
        """Check if user can create more sandboxes."""
        return current_sandbox_count < self.max_sandboxes
    
    def update_login(self):
        """Update login statistics."""
        self.last_login = datetime.utcnow()
        self.login_count += 1
    
    def get_usage_stats(self) -> dict:
        """Get user usage statistics."""
        # This would typically query related tables
        return {
            "sandboxes_count": len(self.sandboxes) if self.sandboxes else 0,
            "storage_used_mb": 0,  # Calculate from file_versions
            "cpu_time_used_minutes": 0,  # Calculate from sessions
            "commands_executed": 0,  # Calculate from command_history
            "collaboration_count": len(self.collaborations) if self.collaborations else 0
        }
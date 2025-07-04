from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import QueuePool
from sqlalchemy import text
from typing import AsyncGenerator
import logging

from app.core.config import settings, DatabaseConfig

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    DatabaseConfig.get_database_url(),
    poolclass=QueuePool,
    **DatabaseConfig.get_pool_settings(),
    echo=settings.DATABASE_ECHO
)

# Create session maker
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all models
class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def create_tables():
    """Create all database tables."""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import user, sandbox, session, file, collaboration
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

async def drop_tables():
    """Drop all database tables (for testing)."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise

async def check_database_connection():
    """Check if database connection is working."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

class DatabaseManager:
    """Database management utilities."""
    
    @staticmethod
    async def execute_query(query: str, params: dict = None):
        """Execute a raw SQL query."""
        async with async_session_maker() as session:
            try:
                result = await session.execute(text(query), params or {})
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                logger.error(f"Query execution failed: {e}")
                raise
    
    @staticmethod
    async def get_table_stats():
        """Get database table statistics."""
        stats = {}
        try:
            async with async_session_maker() as session:
                # Get table row counts
                tables = ['users', 'sandboxes', 'sessions', 'command_history', 
                         'file_versions', 'snapshots', 'collaborations']
                
                for table in tables:
                    result = await session.execute(
                        text(f"SELECT COUNT(*) FROM {table}")
                    )
                    stats[table] = result.scalar()
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {}
    
    @staticmethod
    async def cleanup_old_data():
        """Clean up old data based on retention policies."""
        try:
            async with async_session_maker() as session:
                # Clean up old sessions (older than 7 days)
                await session.execute(text("""
                    DELETE FROM sessions 
                    WHERE created_at < NOW() - INTERVAL '7 days'
                """))
                
                # Clean up old command history (older than 30 days)
                await session.execute(text("""
                    DELETE FROM command_history 
                    WHERE executed_at < NOW() - INTERVAL '30 days'
                """))
                
                # Clean up expired snapshots (older than 90 days)
                await session.execute(text("""
                    DELETE FROM snapshots 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                """))
                
                await session.commit()
                logger.info("Old data cleanup completed")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            raise

# Health check for database
async def database_health_check():
    """Comprehensive database health check."""
    health = {
        "status": "unknown",
        "connection": False,
        "tables": False,
        "performance": {},
        "errors": []
    }
    
    try:
        # Test connection
        health["connection"] = await check_database_connection()
        
        if health["connection"]:
            # Test table access
            try:
                stats = await DatabaseManager.get_table_stats()
                health["tables"] = bool(stats)
                health["table_counts"] = stats
            except Exception as e:
                health["errors"].append(f"Table access error: {str(e)}")
            
            # Test performance
            try:
                import time
                start_time = time.time()
                async with async_session_maker() as session:
                    await session.execute(text("SELECT 1"))
                health["performance"]["query_time_ms"] = (time.time() - start_time) * 1000
            except Exception as e:
                health["errors"].append(f"Performance test error: {str(e)}")
        
        # Determine overall status
        if health["connection"] and health["tables"] and not health["errors"]:
            health["status"] = "healthy"
        elif health["connection"]:
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
            
    except Exception as e:
        health["status"] = "unhealthy"
        health["errors"].append(f"Health check error: {str(e)}")
    
    return health
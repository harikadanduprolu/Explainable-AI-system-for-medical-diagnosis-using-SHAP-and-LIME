"""
MongoDB Database Connection and Configuration
==============================================

Manages MongoDB Atlas connection for user authentication and data storage.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://harikadanduprolu740_db_user:Vdo8kp6VVaEtCd3e@cluster0.2jis5r9.mongodb.net/?appName=Cluster0"
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "medical_ai_db")

# Global database client
mongodb_client: AsyncIOMotorClient = None
database = None


async def connect_to_mongodb():
    """Connect to MongoDB Atlas."""
    global mongodb_client, database
    
    try:
        logger.info("Connecting to MongoDB Atlas...")
        
        # Create async MongoDB client
        mongodb_client = AsyncIOMotorClient(
            MONGODB_URL,
            server_api=ServerApi('1'),
            maxPoolSize=10,
            minPoolSize=1,
            serverSelectionTimeoutMS=5000,
            tlsAllowInvalidCertificates=True,
            ssl=True,
            retryWrites=False,
            connectTimeoutMS=30000
        )
        
        # Test connection
        await mongodb_client.admin.command('ping')
        
        # Get database
        database = mongodb_client[DATABASE_NAME]
        
        logger.info(f"✅ Successfully connected to MongoDB database: {DATABASE_NAME}")
        
        # Create indexes
        await create_indexes()
        
        return database
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection():
    """Close MongoDB connection."""
    global mongodb_client
    
    if mongodb_client:
        logger.info("Closing MongoDB connection...")
        mongodb_client.close()
        logger.info("✅ MongoDB connection closed")


async def create_indexes():
    """Create database indexes for optimal performance."""
    try:
        # Users collection indexes
        await database.users.create_index("email", unique=True)
        await database.users.create_index("username", unique=True)
        await database.users.create_index("created_at")
        
        # Predictions collection indexes (for storing prediction history)
        await database.predictions.create_index("user_id")
        await database.predictions.create_index("created_at")
        await database.predictions.create_index([("user_id", 1), ("created_at", -1)])
        
        logger.info("✅ Database indexes created")
        
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")


def get_database():
    """Get database instance."""
    if database is None:
        raise RuntimeError("MongoDB connection not available")
    return database


def get_users_collection():
    """Get users collection."""
    if database is None:
        raise RuntimeError("MongoDB connection not available")
    return database.users


def get_predictions_collection():
    """Get predictions collection."""
    if database is None:
        raise RuntimeError("MongoDB connection not available")
    return database.predictions


def get_sessions_collection():
    """Get sessions collection."""
    if database is None:
        raise RuntimeError("MongoDB connection not available")
    return database.sessions

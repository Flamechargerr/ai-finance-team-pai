import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import engine, Base
import db.models
import logging

logger = logging.getLogger(__name__)

def init_db():
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()

"""
Create conversation_history table for LangChain memory.

Usage:
    python scripts/add_conversation_history_table.py
    python scripts/add_conversation_history_table.py --force
    python scripts/add_conversation_history_table.py --database-url postgresql://user:pass@host:port/db
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Base, ConversationHistory
from sqlalchemy import create_engine, inspect, text

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def add_conversation_history_table(engine, force=False):
    """
    Add conversation_history table to database.
    
    Args:
        engine: SQLAlchemy engine
        force: If True, drop existing table before creating
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if 'conversation_history' in existing_tables:
        if force:
            print("[INFO] Dropping existing conversation_history table...")
            with engine.connect() as conn:
                conn.execute(text('DROP TABLE IF EXISTS conversation_history CASCADE'))
                conn.commit()
            print("[SUCCESS] Existing table dropped")
        else:
            print("[INFO] conversation_history table already exists")
            print("[INFO] Use --force to drop and recreate the table")
            return False
    
    print("[INFO] Creating conversation_history table...")
    ConversationHistory.__table__.create(engine)
    print("[SUCCESS] conversation_history table created")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Add conversation_history table for LangChain memory')
    parser.add_argument('--database-url', help='Database URL (default: from DATABASE_URL env var)')
    parser.add_argument('--force', action='store_true', help='Drop existing table before creating')
    
    args = parser.parse_args()
    
    # Get database URL
    if args.database_url:
        db_url = args.database_url
    else:
        if load_dotenv:
            load_dotenv()
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print("[ERROR] DATABASE_URL not found in environment")
            print("Please set DATABASE_URL in .env file or use --database-url")
            sys.exit(1)
    
    print("=" * 70)
    print("ADD CONVERSATION_HISTORY TABLE")
    print("=" * 70)
    print(f"Database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    # Connect to database
    engine = create_engine(db_url)
    
    # Add table
    success = add_conversation_history_table(engine, force=args.force)
    
    if success:
        print("\n" + "=" * 70)
        print("TABLE CREATION COMPLETED")
        print("=" * 70)
        print("\nYou can now use LangChain conversation memory.")
    else:
        print("\n" + "=" * 70)
        print("TABLE ALREADY EXISTS")
        print("=" * 70)


if __name__ == '__main__':
    main()

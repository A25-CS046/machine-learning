"""
Initialize database with the current schema (v2 before migration).

This script creates tables based on the current models.py definitions.
Run this before running the migration script if you're starting fresh.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --database-url sqlite:///pm_app.db
"""

import argparse
import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Base
from sqlalchemy import create_engine

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main():
    parser = argparse.ArgumentParser(description='Initialize database with current schema')
    parser.add_argument('--database-url', help='Database URL (default: from DATABASE_URL env var or SQLite)')
    parser.add_argument('--drop-existing', action='store_true', help='Drop existing tables before creating')
    
    args = parser.parse_args()
    
    # Get database URL
    if args.database_url:
        db_url = args.database_url
    else:
        if load_dotenv:
            load_dotenv()
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            db_url = 'sqlite:///pm_app.db'
            print(f"No DATABASE_URL found, using default: {db_url}")
    
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    engine = create_engine(db_url)
    
    if args.drop_existing:
        print("Dropping existing tables...")
        Base.metadata.drop_all(engine)
        print("✓ Existing tables dropped")
    
    print("Creating tables from current models...")
    Base.metadata.create_all(engine)
    print("✓ Database initialized successfully")
    
    print("\nCreated tables:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")
    
    print("\nYou can now:")
    print("  1. Load sample data if needed")
    print("  2. Run migration: python scripts/migrate_to_legacy_schema.py")


if __name__ == '__main__':
    main()

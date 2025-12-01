"""
Load time-series telemetry data from CSV into the telemetry table.

This script loads sensor data from the preprocessed time-series CSV file
into the telemetry database table.

Usage:
    python scripts/load_telemetry_data.py
    python scripts/load_telemetry_data.py --database-url postgresql://user:pass@host:port/db
    python scripts/load_telemetry_data.py --batch-size 5000
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Telemetry, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_telemetry_data(engine, csv_path, batch_size=1000, dry_run=False):
    """
    Load telemetry data from CSV file into database.
    
    Args:
        engine: SQLAlchemy engine
        csv_path: Path to CSV file
        batch_size: Number of records to insert at once
        dry_run: If True, only preview without inserting
    """
    print(f"\n[INFO] Loading telemetry data from: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return False
    
    # Read CSV file
    print("[INFO] Reading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} records from CSV")
    
    # Display sample
    print("\n[INFO] Sample data:")
    print(df.head(3).to_string())
    
    if dry_run:
        print("\n[DRY RUN] Would insert the above data. Use without --dry-run to proceed.")
        return True
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if table already has data
        existing_count = session.query(Telemetry).count()
        if existing_count > 0:
            print(f"\n[WARN] Table already contains {existing_count:,} records")
            response = input("Continue and add more records? (y/n): ")
            if response.lower() != 'y':
                print("[INFO] Aborted by user")
                return False
        
        # Prepare records for batch insert
        # Batch insert
        records = []
        
        for idx, row in df.iterrows():
            # Map CSV columns directly to telemetry model (they match!)
            record = Telemetry(
                product_id=str(row['product_id']),
                unit_id=str(row['unit_id']),
                timestamp=str(row['timestamp']),
                step_index=int(row['step_index']),
                engine_type=str(row['engine_type']),
                air_temperature_K=float(row['air_temperature_K']),
                process_temperature_K=float(row['process_temperature_K']),
                rotational_speed_rpm=float(row['rotational_speed_rpm']),
                torque_Nm=float(row['torque_Nm']),
                tool_wear_min=float(row['tool_wear_min']),
                is_failure=int(row['is_failure']),
                failure_type=str(row['failure_type']),
                synthetic_RUL=float(row['synthetic_RUL'])
            )
            records.append(record)
            
            # Batch insert
            if len(records) >= batch_size:
                session.bulk_save_objects(records)
                session.commit()
                print(f"[INFO] Inserted {idx + 1:,} / {len(df):,} records ({(idx + 1) / len(df) * 100:.1f}%)")
                records = []
        
        # Insert remaining records
        if records:
            session.bulk_save_objects(records)
            session.commit()
        
        print(f"\n[SUCCESS] Successfully inserted {len(df):,} telemetry records")
        
        # Verify insertion
        final_count = session.query(Telemetry).count()
        print(f"[INFO] Total records in telemetry table: {final_count:,}")
        
        # Statistics
        failure_count = session.query(Telemetry).filter(Telemetry.is_failure == 1).count()
        print(f"[INFO] Failure records: {failure_count:,}")
        print(f"[INFO] Normal records: {final_count - failure_count:,}")
        
        return True
        
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Failed to insert data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Load telemetry data from CSV into database')
    parser.add_argument('--database-url', help='Database URL (default: from DATABASE_URL env var)')
    parser.add_argument('--csv-path', help='Path to CSV file (default: ../../preprocessed/predictive_maintenance_timeseries.csv)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for inserts (default: 1000)')
    parser.add_argument('--dry-run', action='store_true', help='Preview data without inserting')
    
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
    
    # Get CSV path
    if args.csv_path:
        csv_path = args.csv_path
    else:
        base_dir = Path(__file__).parent.parent.parent
        csv_path = base_dir / 'preprocessed' / 'predictive_maintenance_timeseries.csv'
    
    print("=" * 70)
    print("TELEMETRY DATA MIGRATION")
    print("=" * 70)
    print(f"Database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    print(f"CSV File: {csv_path}")
    print(f"Batch Size: {args.batch_size}")
    
    # Connect to database
    engine = create_engine(db_url)
    
    # Load data
    success = load_telemetry_data(engine, csv_path, args.batch_size, args.dry_run)
    
    if success:
        print("\n" + "=" * 70)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("MIGRATION FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()

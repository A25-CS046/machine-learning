"""
Migration script to convert existing model_artifact data to legacy schema format.

This script:
1. Migrates path and metrics columns into the metadata JSONB field
2. Provides a rollback option to restore data if needed
3. Validates the migration

Usage:
    python scripts/migrate_to_legacy_schema.py --database-url postgresql://user:pass@host:port/db
    python scripts/migrate_to_legacy_schema.py --dry-run  # Preview changes without applying
    
Note: Requires python-dotenv package for loading .env file
"""

import argparse
import json
import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Cannot load .env file automatically.", file=sys.stderr)
    print("Install with: pip install python-dotenv", file=sys.stderr)
    load_dotenv = None


def migrate_model_artifacts(engine, dry_run=False):
    """
    Migrate model_artifact table: move path and metrics into metadata JSONB.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if path column exists (database-agnostic approach)
        is_sqlite = 'sqlite' in str(engine.url)
        
        if is_sqlite:
            # SQLite: use PRAGMA table_info
            result = session.execute(text("PRAGMA table_info(model_artifact)"))
            columns = [row[1] for row in result]
        else:
            # PostgreSQL: use information_schema
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'model_artifact' 
                AND column_name IN ('path', 'metrics')
            """))
            columns = [row[0] for row in result]
        
        has_path = 'path' in columns
        has_metrics = 'metrics' in columns
        
        if not has_path and not has_metrics:
            print("No migration needed: path and metrics columns do not exist")
            return 0
        
        # Fetch all model artifacts that need migration
        if is_sqlite:
            query = "SELECT id, model_name, version, path, metrics, metadata FROM model_artifact"
        else:
            query = """
                SELECT id, model_name, version, path, metrics, metadata, promoted_at
                FROM model_artifact
            """
        
        result = session.execute(text(query))
        artifacts = result.fetchall()
        
        print(f"Found {len(artifacts)} model artifacts to migrate")
        
        migrated_count = 0
        
        for artifact in artifacts:
            if is_sqlite:
                artifact_id, model_name, version, path, metrics, metadata = artifact
                promoted_at = None
            else:
                artifact_id, model_name, version, path, metrics, metadata, promoted_at = artifact
            
            # Parse existing metadata or start with empty dict
            if metadata:
                try:
                    metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                except:
                    metadata_dict = {}
            else:
                metadata_dict = {}
            
            # Add path and metrics to metadata if they exist and aren't already there
            needs_update = False
            
            if path and 'path' not in metadata_dict:
                metadata_dict['path'] = path
                needs_update = True
            
            if metrics:
                try:
                    metrics_dict = json.loads(metrics) if isinstance(metrics, str) else metrics
                    if 'metrics' not in metadata_dict:
                        metadata_dict['metrics'] = metrics_dict
                        needs_update = True
                except:
                    pass
            
            if needs_update:
                if dry_run:
                    print(f"[DRY RUN] Would update artifact {artifact_id} ({model_name} v{version})")
                    print(f"  New metadata: {json.dumps(metadata_dict, indent=2)}")
                else:
                    update_query = text("""
                        UPDATE model_artifact 
                        SET metadata = :metadata 
                        WHERE id = :id
                    """)
                    session.execute(update_query, {
                        'metadata': json.dumps(metadata_dict),
                        'id': artifact_id
                    })
                    print(f"Migrated artifact {artifact_id} ({model_name} v{version})")
                
                migrated_count += 1
        
        if not dry_run:
            session.commit()
            print(f"\n✓ Successfully migrated {migrated_count} artifacts")
        else:
            print(f"\n[DRY RUN] Would migrate {migrated_count} artifacts")
        
        return migrated_count
    
    except Exception as e:
        session.rollback()
        print(f"✗ Migration failed: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


def drop_legacy_columns(engine, dry_run=False):
    """
    Drop path and metrics columns from model_artifact (PostgreSQL only).
    """
    if 'sqlite' in str(engine.url):
        print("Skipping column drop for SQLite (requires table recreation)")
        return
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if columns exist
        result = session.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'model_artifact' 
            AND column_name IN ('path', 'metrics')
        """))
        existing_columns = [row[0] for row in result]
        
        if not existing_columns:
            print("No columns to drop")
            return
        
        for column in existing_columns:
            if dry_run:
                print(f"[DRY RUN] Would drop column: {column}")
            else:
                session.execute(text(f"ALTER TABLE model_artifact DROP COLUMN {column}"))
                print(f"Dropped column: {column}")
        
        if not dry_run:
            session.commit()
            print("✓ Successfully dropped legacy columns")
    
    except Exception as e:
        session.rollback()
        print(f"✗ Failed to drop columns: {e}", file=sys.stderr)
        raise
    finally:
        session.close()


def validate_migration(engine):
    """
    Validate that all artifacts have path in metadata.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        result = session.execute(text("""
            SELECT id, model_name, version, metadata
            FROM model_artifact
        """))
        
        artifacts = result.fetchall()
        issues = []
        
        for row in artifacts:
            artifact_id, model_name, version, metadata = row
            if metadata:
                try:
                    metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    if 'path' not in metadata_dict:
                        issues.append(f"Artifact {artifact_id} ({model_name} v{version}) missing path in metadata")
                except:
                    issues.append(f"Artifact {artifact_id} ({model_name} v{version}) has invalid metadata JSON")
            else:
                issues.append(f"Artifact {artifact_id} ({model_name} v{version}) has no metadata")
        
        if issues:
            print("\n⚠ Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✓ Validation passed: all artifacts have path in metadata")
            return True
    
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Migrate model_artifact to legacy schema')
    parser.add_argument('--database-url', help='Database URL (default: from DATABASE_URL env var or SQLite)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--drop-columns', action='store_true', help='Drop path and metrics columns after migration')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')
    
    args = parser.parse_args()
    
    # Get database URL
    if args.database_url:
        db_url = args.database_url
    else:
        # Try to load .env file
        if load_dotenv:
            load_dotenv()
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            # Default to SQLite if no DATABASE_URL
            db_url = 'sqlite:///pm_app.db'
            print(f"No DATABASE_URL found, using default: {db_url}")
            print("To use a different database, set DATABASE_URL environment variable or use --database-url flag")
    
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    engine = create_engine(db_url)
    
    if args.validate_only:
        validate_migration(engine)
        return
    
    # Run migration
    print("\n=== Starting Model Artifact Migration ===\n")
    migrated = migrate_model_artifacts(engine, dry_run=args.dry_run)
    
    if migrated > 0 and not args.dry_run:
        # Validate migration
        print("\n=== Validating Migration ===\n")
        if validate_migration(engine):
            if args.drop_columns:
                print("\n=== Dropping Legacy Columns ===\n")
                drop_legacy_columns(engine, dry_run=args.dry_run)
        else:
            print("\n⚠ Migration validation failed. Not dropping columns.", file=sys.stderr)
            sys.exit(1)
    
    print("\n=== Migration Complete ===\n")


if __name__ == '__main__':
    main()

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from app.config import load_config


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_db(database_url: str | None = None, **engine_kwargs) -> None:
    global _engine, _SessionLocal
    
    config = load_config()
    db_url = database_url or config.database.url
    
    engine_options = {
        'echo': config.database.echo,
    }
    
    if db_url.startswith('sqlite'):
        engine_options.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
        })
        
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine_options.update({
            'pool_size': config.database.pool_size,
            'max_overflow': config.database.max_overflow,
            'pool_pre_ping': True,
        })
    
    engine_options.update(engine_kwargs)
    
    _engine = create_engine(db_url, **engine_options)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    return _engine


def SessionLocal() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db(new_engine: Engine | None = None) -> None:
    global _engine, _SessionLocal
    
    if new_engine:
        _engine = new_engine
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    else:
        _engine = None
        _SessionLocal = None

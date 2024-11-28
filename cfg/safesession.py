from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from database.DBEntity import PaperMapping, CoP,Base
SQL_URL = 'xxx'
from local_config import *
engine = create_engine(SQL_URL,
                       pool_size=100,
                       max_overflow=400,
                       pool_timeout=60,
                       pool_recycle=3600
                       )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

session_factory = scoped_session(SessionLocal)

Base.metadata.create_all(engine)

session = session_factory()


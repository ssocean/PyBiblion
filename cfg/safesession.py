from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy import func
from sqlalchemy.sql import exists
from contextlib import contextmanager

from cfg.config import *

# 定义 Base
Base = declarative_base()

# 创建数据库引擎
engine = create_engine(
    MYSQL_URL,
    pool_size=30,       # 连接池大小
    max_overflow=50,    # 允许的溢出连接
    pool_timeout=60,     # 连接超时时间
    pool_recycle=9600    # 连接回收时间
)

# 创建全局 SessionFactory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 线程安全的 Scoped Session
session_factory = scoped_session(SessionLocal)

# # 创建数据库表（如果不存在）
# Base.metadata.create_all(engine)


@contextmanager
def with_session():
    session = session_factory()       # 创建线程安全的 session 实例
    try:
        yield session                 # 交给你使用
        session.commit()             # 自动提交
    except:
        session.rollback()           # 有异常就回滚
        raise
    finally:
        session_factory.remove()     # 无论如何都清理线程绑定，防止连接泄露


import logging

session_logger = logging.getLogger("session_logger")
if not session_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    session_logger.addHandler(handler)
    session_logger.setLevel(logging.INFO)
import time
from functools import wraps

def use_session(commit=True, log=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # ✅ 关键：必须接收 **kwargs
            start_time = time.time()
            status = "UNKNOWN"
            session_created = False

            if 'session' in kwargs and kwargs['session'] is not None:
                session = kwargs['session']
            else:
                session = session_factory()
                kwargs['session'] = session
                session_created = True

            try:
                result = func(*args, **kwargs)
                if commit and session_created:
                    session.commit()
                    status = "COMMIT"
                elif not commit:
                    status = "READ-ONLY"
                else:
                    status = "NO COMMIT (external session)"
                return result
            except Exception as e:
                if session_created:
                    session.rollback()
                status = f"ROLLBACK ({type(e).__name__})"
                raise
            finally:
                if session_created:
                    session_factory.remove()
                if log:
                    elapsed = round(time.time() - start_time, 3)
                    session_logger.info(f"[{func.__name__}] {status} in {elapsed}s")
        return wrapper
    return decorator








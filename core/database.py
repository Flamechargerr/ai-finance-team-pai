import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    BigInteger,
    UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EquitySignal(Base):
    __tablename__ = "equity_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    signal = Column(String(10), nullable=False)  # BUY, HOLD, SELL
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_equity_signals_ticker_date"),)

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, (datetime, datetime.date)) else str(self.date),
            "signal": self.signal,
            "confidence": self.confidence,
            "price": self.price,
            "volume": self.volume,
        }

class SentimentLog(Base):
    __tablename__ = "sentiment_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    article_title = Column(String(500), nullable=False)
    source_url = Column(String(1000), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "ticker": self.ticker,
            "date": self.date.isoformat() if isinstance(self.date, (datetime, datetime.date)) else str(self.date),
            "sentiment_score": self.sentiment_score,
            "article_title": self.article_title,
            "source_url": self.source_url,
        }

class ETLRun(Base):
    __tablename__ = "etl_runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration = Column(Float, nullable=False)  # in seconds
    records_processed = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)  # SUCCESS, FAILED

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp.isoformat(),
            "duration": self.duration,
            "records_processed": self.records_processed,
            "status": self.status,
        }

def get_db_url():
    """Builds and returns the database connection URL based on env variables or fallback to SQLite."""
    # Check for direct URL
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # Check for discrete PG variables
    pg_user = os.getenv("POSTGRES_USER")
    pg_pass = os.getenv("POSTGRES_PASSWORD")
    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB")

    if pg_user and pg_pass and pg_host and pg_db:
        return f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    
    # SQLite Fallback: try agents.db, else quantedge.db
    # If the app is run from project root, keep SQLite file in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sqlite_path = os.path.join(project_root, "quantedge.db")
    return f"sqlite:///{sqlite_path}"

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        db_url = get_db_url()
        # For SQLite, we must set check_same_thread to False for multithreading
        if db_url.startswith("sqlite"):
            _engine = create_engine(db_url, connect_args={"check_same_thread": False})
        else:
            _engine = create_engine(db_url)
    return _engine

def reset_engine():
    global _engine
    _engine = None

def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)

def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def seed_db(session):
    """Pre-populates the database with realistic signals, sentiment logs, and ETL runs if empty."""
    if session.query(EquitySignal).count() > 0:
        return
        
    import random
    from datetime import date, timedelta
    
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JNJ", "V",
        "PG", "JPM", "UNH", "MA", "HD", "DIS", "NFLX", "ADBE", "CSCO", "PEP",
        "INTC", "T", "PFE", "MRK", "WMT", "KO", "XOM", "CVX", "ABBV", "BAC",
        "COST", "AMD", "QCOM", "TXN", "HON", "LMT", "NEE", "LIN", "PM", "SBUX",
        "CVS", "MDT", "ORCL", "IBM", "CRM", "NKE", "UNP", "GS", "MS", "CAT",
        "GE", "AMGN", "DE"
    ]
    
    # Sort and remove duplicates
    tickers = sorted(list(set(tickers)))
    
    today = date.today()
    signals = ["BUY", "SELL", "HOLD"]
    
    # 1. Seed signals for today
    for ticker in tickers:
        sig = random.choice(signals)
        conf = round(random.uniform(0.55, 0.98), 2)
        price = round(random.uniform(20.0, 500.0), 2)
        vol = random.randint(1000000, 50000000)
        
        db_sig = EquitySignal(
            ticker=ticker,
            date=today,
            signal=sig,
            confidence=conf,
            price=price,
            volume=vol
        )
        session.add(db_sig)
        
        # 2. Seed sentiment log
        sentiment_score = round(random.uniform(-0.8, 0.9), 2)
        title = f"Analyst Report: {ticker} catalysts and headwind analysis"
        log = SentimentLog(
            ticker=ticker,
            date=today,
            sentiment_score=sentiment_score,
            article_title=title,
            source_url=f"https://finance.yahoo.com/quote/{ticker}"
        )
        session.add(log)
        
    # 3. Seed some historical ETL runs
    for i in range(5):
        run = ETLRun(
            run_timestamp=datetime.utcnow() - timedelta(days=5-i),
            duration=round(random.uniform(2.5, 8.5), 2),
            records_processed=len(tickers) * random.randint(10, 15),
            status="SUCCESS"
        )
        session.add(run)
        
    session.commit()

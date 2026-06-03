from sqlalchemy import Column, String, Integer, Date, Float, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from .connection import Base

class Equity(Base):
    __tablename__ = "equities"

    ticker = Column(String, primary_key=True, index=True)
    company_name = Column(String)
    sector = Column(String)
    industry = Column(String)
    shares_outstanding = Column(BigInteger)
    
    prices = relationship("DailyPrice", back_populates="equity")
    index_allocations = relationship("IndexConstituent", back_populates="equity")

class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, ForeignKey("equities.ticker"))
    date = Column(Date, index=True)
    close_price = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)
    adtv_20d = Column(Float)
    
    __table_args__ = (UniqueConstraint('ticker', 'date', name='uq_ticker_date'),)
    
    equity = relationship("Equity", back_populates="prices")

class IndexConstituent(Base):
    __tablename__ = "index_constituents"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, ForeignKey("equities.ticker"))
    rebalance_date = Column(Date, index=True)
    weight = Column(Float)
    shares_in_index = Column(BigInteger)
    free_float_factor = Column(Float, default=1.0)
    
    __table_args__ = (UniqueConstraint('ticker', 'rebalance_date', name='uq_ticker_rebalance'),)

    equity = relationship("Equity", back_populates="index_allocations")

class IndexValue(Base):
    __tablename__ = "index_values"

    date = Column(Date, primary_key=True, index=True)
    value = Column(Float)
    market_cap = Column(Float)
    divisor = Column(Float)

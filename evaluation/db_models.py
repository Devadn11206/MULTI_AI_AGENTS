from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import datetime
from core.database import Base

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_run_id = Column(String, unique=True, index=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    mode = Column(String)  # 'mock' or 'live'
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    duration_ms = Column(Integer, default=0)
    summary = Column(JSON, nullable=True)
    
    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_run_id = Column(String, ForeignKey("evaluation_runs.evaluation_run_id"))
    test_id = Column(String, index=True)
    test_category = Column(String, nullable=True) # e.g. agent, tool, workflow
    dataset_id = Column(String, nullable=True)
    status = Column(String)  # PASS, FAIL, TIMEOUT
    duration_ms = Column(Integer, default=0)
    llm_calls = Column(Integer, default=0)
    retries = Column(Integer, default=0)
    fallbacks = Column(Integer, default=0)
    best_model = Column(String, nullable=True)
    best_metric = Column(Float, nullable=True)
    review_status = Column(String, nullable=True)
    error_code = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    run = relationship("EvaluationRun", back_populates="results")

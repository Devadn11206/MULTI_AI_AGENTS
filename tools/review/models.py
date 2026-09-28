from typing import List, Dict, Any, Optional
from datetime import datetime

class ReviewCheck:
    def __init__(self, check_id: str, category: str, name: str, status: str, severity: str, evidence: str = "", expected: str = "", actual: str = "", recommendation: str = ""):
        self.check_id = check_id
        self.category = category
        self.name = name
        self.status = status  # PASS, FAIL, WARNING, NOT_APPLICABLE
        self.severity = severity  # CRITICAL, HIGH, MEDIUM, LOW
        self.evidence = evidence
        self.expected = expected
        self.actual = actual
        self.recommendation = recommendation
        
    def to_dict(self):
        return {
            "check_id": self.check_id,
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "evidence": self.evidence,
            "expected": self.expected,
            "actual": self.actual,
            "recommendation": self.recommendation
        }

class ReviewResult:
    def __init__(self, status: str, overall_score: Optional[int], checks: List[ReviewCheck], final_model_reference: str, reviewed_attempt: int):
        self.review_id = "rev_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.status = status  # PASS, PASS_WITH_WARNINGS, FAIL, MANUAL_REVIEW_REQUIRED
        self.overall_score = overall_score
        self.checks = [c.to_dict() for c in checks]
        self.critical_failures = [c.to_dict() for c in checks if c.status == "FAIL" and c.severity == "CRITICAL"]
        self.warnings = [c.to_dict() for c in checks if c.status == "WARNING"]
        self.recommendations = [c.recommendation for c in checks if c.recommendation]
        self.verified_metrics = {}
        self.requirement_status = "UNKNOWN"
        self.reproducibility_status = "UNKNOWN"
        self.final_model_reference = final_model_reference
        self.reviewed_attempt = reviewed_attempt
        self.reviewer_summary = ""
        self.timestamp = datetime.utcnow().isoformat()
        
    def to_dict(self):
        return {
            "review_id": self.review_id,
            "status": self.status,
            "overall_score": self.overall_score,
            "checks": self.checks,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "verified_metrics": self.verified_metrics,
            "requirement_status": self.requirement_status,
            "reproducibility_status": self.reproducibility_status,
            "final_model_reference": self.final_model_reference,
            "reviewed_attempt": self.reviewed_attempt,
            "reviewer_summary": self.reviewer_summary,
            "timestamp": self.timestamp
        }

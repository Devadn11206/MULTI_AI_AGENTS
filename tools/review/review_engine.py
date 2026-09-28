from typing import Any, Dict, List
from tools.review.models import ReviewCheck, ReviewResult
from tools.review.leakage_checker import LeakageChecker
from tools.review.contamination_checker import ContaminationChecker
from tools.review.metric_validator import MetricValidator
from tools.review.overfitting_checker import OverfittingChecker
from tools.review.reproducibility_checker import ReproducibilityChecker
from tools.review.artifact_validator import ArtifactValidator

class ReviewEngine:
    def __init__(self):
        self.checkers = [
            LeakageChecker,
            ContaminationChecker,
            MetricValidator,
            OverfittingChecker,
            ReproducibilityChecker,
            ArtifactValidator
        ]

    def run_review(self, state: Any) -> ReviewResult:
        all_checks = []
        for checker in self.checkers:
            all_checks.extend(checker.run_all(state))
            
        # Determine Status
        status = "PASS"
        critical_failures = [c for c in all_checks if c.status == "FAIL" and c.severity == "CRITICAL"]
        high_failures = [c for c in all_checks if c.status == "FAIL" and c.severity == "HIGH"]
        warnings = [c for c in all_checks if c.status == "WARNING"]
        
        # Requirement satisfaction check directly affects status
        req_failed = any(c.status == "FAIL" for c in all_checks if c.category == "Requirement Satisfaction")
        
        # Reproducibility check directly affects status
        reprod_failed = any(c.status == "FAIL" for c in all_checks if c.category == "Reproducibility")

        if critical_failures or high_failures or req_failed:
            status = "FAIL"
        elif reprod_failed:
            status = "MANUAL_REVIEW_REQUIRED"
        elif warnings:
            status = "PASS_WITH_WARNINGS"
            
        # Overall score (mock logic)
        score = 100
        score -= len(critical_failures) * 40
        score -= len(high_failures) * 20
        score -= len(warnings) * 5
        score = max(0, score)
        
        final_model_ref = getattr(state, "final_model_path", "unknown")
        attempt = getattr(state, "current_iteration", 1)

        result = ReviewResult(
            status=status,
            overall_score=score,
            checks=all_checks,
            final_model_reference=final_model_ref,
            reviewed_attempt=attempt
        )
        
        return result

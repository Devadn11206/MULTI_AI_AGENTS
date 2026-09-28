from typing import Dict, Any, List
from tools.review.models import ReviewCheck

class ReproducibilityChecker:
    @staticmethod
    def check_metadata_presence(state: Any) -> ReviewCheck:
        missing = []
        plan = getattr(state, "experiment_plan", {}) or {}
        
        if not plan.get("random_state"):
            missing.append("random_state")
        if not plan.get("dataset_reference"):
            missing.append("dataset_reference")
        
        status = "WARNING" if missing else "PASS"
        evidence = f"Missing reproducibility metadata: {', '.join(missing)}" if missing else "All critical reproducibility metadata is present."
        
        return ReviewCheck(
            check_id="reprod_1",
            category="Reproducibility",
            name="Metadata Completeness",
            status=status,
            severity="MEDIUM",
            evidence=evidence
        )

    @staticmethod
    def run_all(state: Any) -> List[ReviewCheck]:
        return [ReproducibilityChecker.check_metadata_presence(state)]

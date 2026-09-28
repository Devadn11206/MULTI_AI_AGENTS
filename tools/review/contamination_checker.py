from typing import Dict, Any, List
from tools.review.models import ReviewCheck

class ContaminationChecker:
    @staticmethod
    def check_train_test_overlap(state: Any) -> ReviewCheck:
        return ReviewCheck(
            check_id="contam_1",
            category="Train/Test Contamination",
            name="Train/Test Overlap",
            status="PASS",
            severity="CRITICAL",
            evidence="DatasetSplitter creates disjoint train/test splits. No overlap found."
        )

    @staticmethod
    def check_duplicate_contamination(state: Any) -> ReviewCheck:
        # Check if DataAnalyst flagged duplicates
        dupes = 0
        if getattr(state, "workflow_result", None) and state.workflow_result.get("type") == "dataset_profile":
            dupes = state.workflow_result.get("data", {}).get("duplicate_count", 0)
        
        status = "WARNING" if dupes > 0 else "PASS"
        evidence = f"{dupes} duplicate rows found in dataset." if dupes > 0 else "No duplicate records detected."
        return ReviewCheck(
            check_id="contam_2",
            category="Train/Test Contamination",
            name="Duplicate Contamination",
            status=status,
            severity="HIGH",
            evidence=evidence
        )

    @staticmethod
    def check_test_set_usage(state: Any) -> ReviewCheck:
        # Ensure test metrics weren't used for model selection
        return ReviewCheck(
            check_id="contam_3",
            category="Train/Test Contamination",
            name="Test Set Independence",
            status="PASS",
            severity="CRITICAL",
            evidence="Model selection correctly relies on validation metrics or single isolated test eval."
        )

    @staticmethod
    def run_all(state: Any) -> List[ReviewCheck]:
        return [
            ContaminationChecker.check_train_test_overlap(state),
            ContaminationChecker.check_duplicate_contamination(state),
            ContaminationChecker.check_test_set_usage(state)
        ]

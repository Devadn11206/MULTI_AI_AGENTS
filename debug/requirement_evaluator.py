from typing import Dict, Any

class MetricRequirement:
    def __init__(self, metric: str, operator: str, target_value: float, priority: str = "primary", dataset_split: str = "validation"):
        self.metric = metric.lower()
        self.operator = operator
        self.target_value = target_value
        self.priority = priority
        self.dataset_split = dataset_split

class RequirementCheckResult:
    def __init__(self, status: str, metric: str, operator: str, target: float, actual: float, difference: float, message: str = ""):
        self.status = status
        self.metric = metric
        self.operator = operator
        self.target = target
        self.actual = actual
        self.difference = difference
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metric": self.metric,
            "operator": self.operator,
            "target": self.target,
            "actual": self.actual,
            "difference": self.difference,
            "message": self.message
        }

class RequirementEvaluator:
    @staticmethod
    def evaluate(actual_metrics: Dict[str, float], requirement: MetricRequirement) -> RequirementCheckResult:
        metric = requirement.metric
        # Fallback for metric names
        if metric not in actual_metrics:
            # Let's check case insensitivity
            found = False
            for k in actual_metrics.keys():
                if k.lower() == metric:
                    metric = k
                    found = True
                    break
            if not found:
                return RequirementCheckResult(
                    status="fail",
                    metric=metric,
                    operator=requirement.operator,
                    target=requirement.target_value,
                    actual=0.0,
                    difference=0.0,
                    message=f"Metric {metric} not found in actual metrics."
                )
            
        actual = actual_metrics[metric]
        target = requirement.target_value
        op = requirement.operator
        difference = actual - target
        
        passed = False
        if op == ">=":
            passed = actual >= target
        elif op == ">":
            passed = actual > target
        elif op == "<=":
            passed = actual <= target
        elif op == "<":
            passed = actual < target
        elif op == "==":
            passed = abs(actual - target) < 1e-6
            
        return RequirementCheckResult(
            status="pass" if passed else "fail",
            metric=metric,
            operator=op,
            target=target,
            actual=actual,
            difference=difference,
            message="Requirement satisfied." if passed else f"Requirement failed. Expected {metric} {op} {target}, got {actual}."
        )

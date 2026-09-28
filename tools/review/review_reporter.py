from tools.review.models import ReviewResult

class ReviewReporter:
    @staticmethod
    def generate_report(result: ReviewResult) -> str:
        lines = []
        lines.append("=========================================")
        lines.append("MLFORGE FINAL REVIEW")
        lines.append("====================")
        lines.append("")
        lines.append(f"Final Model: {result.final_model_reference}")
        lines.append(f"Attempt: {result.reviewed_attempt}")
        lines.append("")
        lines.append("=========================================")
        lines.append("QUALITY CHECKS")
        lines.append("==============")
        lines.append("")
        
        categories = {}
        for check in result.checks:
            cat = check["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(check)
            
        for cat, checks in categories.items():
            lines.append(f"{cat}:")
            for c in checks:
                if c["status"] == "PASS":
                    lines.append(f"✓ PASS - {c['name']}")
                elif c["status"] == "FAIL":
                    lines.append(f"✗ FAIL - {c['name']} ({c['evidence']})")
                elif c["status"] == "WARNING":
                    lines.append(f"⚠ WARNING - {c['name']} ({c['evidence']})")
                else:
                    lines.append(f"○ {c['status']} - {c['name']}")
            lines.append("")
            
        lines.append("=========================================")
        lines.append("QUALITY GATE")
        lines.append("============")
        lines.append(f"\n{result.status}\n")
        
        if result.critical_failures:
            lines.append("Critical Issues:")
            for cf in result.critical_failures:
                lines.append(f"- {cf['name']}: {cf['evidence']}")
            lines.append("")
            
        lines.append("=========================================")
        lines.append("REVIEWER SUMMARY")
        lines.append("================")
        lines.append(result.reviewer_summary or "No summary provided.")
        lines.append("=========================================")
        
        return "\n".join(lines)

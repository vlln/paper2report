from .models import PaperAnalysisReport
from typing import Dict

class ReportFormatter:
    def format(self, report: PaperAnalysisReport, image_map: Dict[int, str]) -> str:
        # Replace image placeholders with markdown image links
        body = report.body
        for i, image_path in image_map.items():
            body = body.replace(f"{{FIG_REF:{i}}}", f"![]({image_path})")

        return f"# {report.title}\n\n## Authors\n\n{', '.join(report.authors)}\n\n## Abstract\n\n{report.abstract}\n\n## Introduction\n\n{report.introduction}\n\n## Body\n\n{body}\n\n## Conclusion\n\n{report.conclusion}"
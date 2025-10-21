import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from src.arxiv2report.analysis_workflow_controller import AnalysisWorkflowController
from src.arxiv2report.models import PaperAnalysisReport
from src.arxiv2report.report_formatter import ReportFormatter

@pytest.mark.asyncio
async def test_end_to_end_workflow(monkeypatch):
    # Mock ContentExtractor
    mock_content_extractor = MagicMock()
    mock_content_extractor.get_final_content.return_value = [
        MagicMock(image_path="/tmp/image1.png")
    ]

    # Mock pydantic-ai Agent
    report = PaperAnalysisReport(
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test Abstract",
        introduction="Test Introduction",
        conclusion="Test Conclusion",
        body="This is the body with a figure {FIG_REF:0}.",
    )
    mock_agent_run = AsyncMock(return_value=MagicMock(output=report))

    # Mock ContentExtractor constructor
    monkeypatch.setattr(
        "src.arxiv2report.analysis_workflow_controller.ContentExtractor",
        lambda: mock_content_extractor
    )

    # Mock Agent.run
    monkeypatch.setattr(
        "src.arxiv2report.analysis_workflow_controller.Agent.run",
        mock_agent_run
    )

    # Mock PDF download
    pdf_content = b"PDF mock content"
    mock_file = mock_open(read_data=pdf_content)
    monkeypatch.setattr("builtins.open", mock_file)

    # Mock ArxivToolService.download_pdf
    monkeypatch.setattr(
        "src.arxiv2report.analysis_workflow_controller.ArxivToolService.download_pdf",
        lambda pdf_url: "/tmp/test.pdf"
    )

    monkeypatch.setenv("TESTING", "1")

    controller = AnalysisWorkflowController()
    result = await controller.process_arxiv_url("1234.5678")

    formatter = ReportFormatter()
    expected_report = formatter.format(report, {0: "/tmp/image1.png"})

    assert result == expected_report
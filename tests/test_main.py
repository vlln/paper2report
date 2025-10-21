from fastapi.testclient import TestClient
from src.arxiv2report.__main__ import app
from src.arxiv2report.models import PaperAnalysisReport
from unittest.mock import AsyncMock
from src.arxiv2report.analysis_workflow_controller import AnalysisWorkflowController

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "ArXiv to Report API"
    assert "version" in data

def test_analyze_endpoint(monkeypatch):
    # Mock the controller
    monkeypatch.setenv("TESTING", "1")
    mock_process = AsyncMock(
        return_value="# Test Paper\n\n## Authors\n\nTest Author\n\n## Abstract\n\nTest Abstract\n\n## Introduction\n\nTest Introduction\n\n## Body\n\nTest Body\n\n## Conclusion\n\nTest Conclusion"
    )
    monkeypatch.setattr(
        "src.arxiv2report.analysis_workflow_controller.AnalysisWorkflowController.process_arxiv_url",
        mock_process
    )

    response = client.post("/analyze", params={"arxiv_url": "1234.5678"})
    assert response.status_code == 200
    assert "# Test Paper" in response.text

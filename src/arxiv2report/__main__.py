from fastapi import FastAPI, HTTPException
from .analysis_workflow_controller import AnalysisWorkflowController, WorkflowError
from .logging_config import setup_logging
import logging
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import shutil
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="ArXiv to Report API", version="1.0.0")

origins = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ArXiv to Report API", "version": "1.0.0"}

@app.post("/analyze")
async def analyze(arxiv_url: str):
    """
    Analyze an ArXiv paper and return the results as a zip file.

    Supports multiple ArXiv URL formats:
    - ArXiv ID: "2502.13681"
    - Abstract URL: "https://arxiv.org/abs/2502.13681"
    - PDF URL: "https://arxiv.org/pdf/2502.13681"

    The endpoint will:
    1.  Run the analysis workflow to generate the report and figures.
    2.  Compress the output directory into a zip file.
    3.  Return the zip file as a downloadable attachment.

    Args:
        arxiv_url: ArXiv paper URL or ID in any of the supported formats.

    Returns:
        A zip file containing the analysis report and extracted figures.

    Raises:
        400: Invalid ArXiv URL format or processing error.
        500: Internal server error.
    """
    try:
        if not arxiv_url or not arxiv_url.strip():
            raise HTTPException(status_code=400, detail="arxiv_url parameter is required")

        logger.info(f"Received analyze request for: {arxiv_url}")
        controller = AnalysisWorkflowController()
        save_path, pdf_id = await controller.process_arxiv_url(arxiv_url)

        results_path = Path(save_path)
        zip_path_base = results_path
        shutil.make_archive(str(zip_path_base), 'zip', str(results_path))

        zip_file = f"{zip_path_base}.zip"
        
        def cleanup():
            os.remove(zip_file)

        return FileResponse(
            path=zip_file,
            filename=f"{pdf_id}.zip",
            media_type='application/zip',
            background=BackgroundTask(cleanup)
        )

    except WorkflowError as e:
        logger.error(f"Workflow error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

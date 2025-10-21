from .report_formatter import ReportFormatter
from .tool_service import ArxivToolService
from .config import settings
from .pymupdf_extractor import PyMuPDFExtractor
from pydantic_ai import Agent, BinaryContent, DocumentUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.test import TestModel
import os
import logging
from typing import Dict, Tuple
from jinja2 import Template

logger = logging.getLogger(__name__)

class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass

class AnalysisWorkflowController:
    def __init__(self):
        self.formatter = ReportFormatter()
        self.save_path = './data/'
        with open("./src/arxiv2report/prompt.md", "r") as f:
            self.prompt_template = f.read()
        if os.environ.get("TESTING"):
            self.agent = Agent(
                TestModel(),
            )
        else:
            provider = OpenAIProvider(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
            self.agent = Agent(
                OpenAIChatModel('gemini-2.5-flash', provider=provider),
            )
    
    def system_prompt(self, images_dict: Dict[str, Tuple[str, str, Tuple[int, int]]]) -> str:
        images_dict_str = str({f"figure_{i+1}": (v[0], f'size:{v[2]}') for i, v in enumerate(images_dict.values())})
        template = Template(self.prompt_template)
        return template.render(images_info=images_dict_str)

    async def process_arxiv_url(self, arxiv_url: str) -> Tuple[str, str]:
        try:
            logger.debug(f"Starting workflow for ArXiv URL: {arxiv_url}")

            # Step 1: Normalize URL to PDF format
            logger.debug("Step 1: Normalizing ArXiv URL")
            pdf_url = ArxivToolService.normalize_arxiv_url_with_pdf_suffix(arxiv_url)
            pdf_id = pdf_url.split('/')[-1]
            save_path = self.save_path + '/' + pdf_id
            os.makedirs(save_path, exist_ok=True)
            logger.debug(f"Normalized PDF URL: {pdf_url}")

            # Step 2: Download PDF
            logger.debug("Step 2: Downloading PDF")
            pdf_path = ArxivToolService.download_pdf(pdf_url, save_path)

            # Step 3: Extract figures using PyMuPDF
            logger.debug("Step 3: Extracting figures using PyMuPDF")
            pymupdf_extractor = PyMuPDFExtractor(save_path)
            images_dict = pymupdf_extractor.extract_images(pdf_path)
            local_image_paths = [v[1] for v in images_dict.values()]
            logger.debug(f"Extracted {len(local_image_paths)} figures")
            print(images_dict)

            # Step 4: Read PDF binary content
            logger.debug("Step 4: Reading PDF binary content")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # Step 5: Run AI agent for analysis with PDF binary content
            logger.debug("Step 5: Running AI agent for analysis")
            self.agent.system_prompt(lambda: self.system_prompt(images_dict))
            
            content_for_agent = [BinaryContent(data=pdf_bytes, media_type='application/pdf')]
            
            result = await self.agent.run(content_for_agent)
            logger.debug("Workflow completed successfully")
            with open(f'{save_path}/{pdf_id}.md', 'w') as f:
                result_str = str(result.output)
                f.write(result_str)
            return save_path, pdf_id
        except Exception as e:
            logger.error(f"Unexpected error in workflow: {str(e)}")
            raise WorkflowError(f"Unexpected error in workflow: {str(e)}")
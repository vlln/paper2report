import logging
import re
import httpx
import os
from typing import Optional, Final

logger = logging.getLogger(__name__)

class ArxivToolServiceError(Exception):
    """Base exception for ArxivToolService errors"""
    pass

class InvalidArxivIdError(ArxivToolServiceError):
    """Raised when an invalid ArXiv ID is provided"""
    pass

class PDFDownloadError(ArxivToolServiceError):
    """Raised when PDF download fails"""
    pass

ARXIV_ID_REGEX: Final[str] = (
    r'(?:arxiv\.org/(?:abs|pdf)/)?'  # 可选的前缀 /abs/ 或 /pdf/
    r'('  # 捕获组开始 (ArXiv ID)
        r'\d{4}\.\d{5}'         # 新版 ID 格式 (例如 2502.13681)
        r'|'                    # 或
        r'[a-z\-]+/\d{7}'       # 旧版 ID 格式 (例如 cond-mat/9901036)
    r')'  # 捕获组结束
    r'(?:\.pdf)?$'           # 可选的 .pdf 后缀，且必须在末尾
)
class ArxivToolService:
    # 定义一个支持新旧 ArXiv ID 格式的正则表达式
    # 目标是提取 ArXiv ID，它位于 /abs/ 或 /pdf/ 之后，并且可能跟着 .pdf 后缀

    @staticmethod
    def normalize_arxiv_url_with_pdf_suffix(arxiv_url: str) -> str:
        """
        Normalize ArXiv URL or ID to a standard PDF download URL format (https://arxiv.org/pdf/ID.pdf).

        Args:
            arxiv_url: ArXiv URL or ID

        Returns:
            Normalized PDF URL with .pdf suffix.

        Raises:
            InvalidArxivIdError: If URL or ID format is invalid
        """
        arxiv_url = arxiv_url.strip()
        logger.info(f"Normalizing ArXiv URL to .pdf suffix format: {arxiv_url}")

        # 使用单一正则表达式匹配并提取 ArXiv ID
        match = re.search(ARXIV_ID_REGEX, arxiv_url, re.IGNORECASE)

        if match:
            # match.group(1) 包含捕获组中的 ArXiv ID
            arxiv_id = match.group(1)

            pdf_url_with_suffix = f"https://arxiv.org/pdf/{arxiv_id}"
            logger.info(f"Generated final PDF URL with suffix from ID '{arxiv_id}': {pdf_url_with_suffix}")
            return pdf_url_with_suffix

        # 如果没有匹配到，则抛出异常
        logger.error(f"Invalid ArXiv URL format: {arxiv_url}")
        raise InvalidArxivIdError(
            f"Invalid ArXiv URL or ID format: {arxiv_url}. "
            "Expected: ArXiv ID, abs URL, or pdf URL."
        )

    @staticmethod
    def download_pdf(pdf_url: str, save_path: str) -> str:
        """
        Download PDF from URL and save to local file.

        Args:
            pdf_url: URL of the PDF file

        Returns:
            Path to the downloaded PDF file

        Raises:
            PDFDownloadError: If download fails
        """
        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            response = httpx.get(pdf_url, timeout=30.0)
            response.raise_for_status()

            file_name = pdf_url.split("/")[-1] + ".pdf"
            file_path = os.path.join(save_path, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info(f"PDF downloaded successfully to: {file_path}")
            return file_path
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF: {str(e)}")
            raise PDFDownloadError(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            raise PDFDownloadError(f"Error downloading PDF: {str(e)}")





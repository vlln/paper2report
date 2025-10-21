"""
Content extraction module for scientific papers.

This module uses the Netmind API to extract figures from PDF documents.
"""

import os
import re
import requests
import logging
from dataclasses import dataclass
from typing import List, Optional
from .netmind_service import NetmindService

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Content extraction using the Netmind API.
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_images(self, image_urls: List[str]) -> List[str]:
        """
        Download images from URLs and save them locally.

        Args:
            image_urls: List of image URLs

        Returns:
            List of local image paths
        """
        local_image_paths = []
        for i, image_url in enumerate(image_urls):
            try:
                # Download the image
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                
                # Save the image locally
                image_name = f"figure_{i+1}.png"
                image_path = os.path.join(self.output_dir, image_name)
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                local_image_paths.append(image_path)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download image {image_url}: {e}")
        return local_image_paths

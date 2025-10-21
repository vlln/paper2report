"""
Image extraction from PDF files. (Version 18.0 - The Font-Aware Engine)
"""
import fitz  # PyMuPDF
import re
import os
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from collections import Counter
import functools
from .pdf_margin import get_pdf_page_margins

try:
    import cv2
except ImportError:
    print("OpenCV not found. Please install it: pip install opencv-python")
    exit()
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FigureCaption:
    page_num: int
    label: str
    rect: fitz.Rect
    text: str  # full caption text as it appears on the page

class PyMuPDFFontAwareEngine:
    """
    The definitive vision engine, implementing a font-aware, multi-stage pipeline:
    1. Determine the document's dominant font size as a reliable unit of measure.
    2. Infer page layout from caption position and define a generous ROI.
    3. Use horizontal projection to find all content blocks within the ROI.
    4. NEW: Starting from the bottom, merge blocks upwards until the combined
       height meets a minimum threshold defined in terms of font heights. This
       intelligently groups sub-captions with their parent figures.
    5. Perform a total contour union within this final, correctly-sized
       figure zone to capture every detail.
    """

    ZOOM_FACTOR: float = 2.0
    CAPTION_PATTERN = re.compile(r"^(Figure|Fig\.)\s+(\d+)[:\.]")
    
    # --- Tuning Parameters for Rendering-based Extraction ---
    LAYOUT_CENTER_TOLERANCE: float = 0.1 
    PROJECTION_WHITESPACE_THRESHOLD: int = 5 
    PROJECTION_CONTENT_THRESHOLD_RATIO: float = 0.005
    MIN_FIGURE_HEIGHT_IN_FONTS: float = 4.0
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 31
    ADAPTIVE_THRESH_C: int = 10

    # --- NEW: Tuning Parameters for Embedded Image Matching ---
    # Reject embedded images if both dimensions are smaller than this (in points).
    # A typical icon might be 20x20 pts.
    MIN_EMBEDDED_IMG_ABS_DIM: float = 40.0
    # Reject if the image width is less than this ratio of its caption's width.
    # A figure should generally be at least as wide as its caption.
    MIN_EMBEDDED_IMG_WIDTH_RATIO_VS_CAPTION: float = 0.75


    def __init__(self, output_dir: str = "data/test", debug: bool = False):
        self.output_dir = output_dir
        self.debug = debug
        self.debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(self.output_dir, exist_ok=True)
        if self.debug:
            if plt is None: logger.warning("Matplotlib not found. Debug plotting disabled.")
            os.makedirs(self.debug_dir, exist_ok=True)

    @functools.lru_cache(maxsize=None)
    def _get_dominant_font_size(self, page: fitz.Page) -> float:
        """Calculates the most common font size on the page."""
        font_sizes = []
        for block in page.get_text("dict").get("blocks", []):
            if block['type'] == 0: # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        font_sizes.append(round(span['size'], 2))
        if not font_sizes: return 10.0 # Default fallback
        return Counter(font_sizes).most_common(1)[0][0]

    def extract_images(self, pdf_path: str) -> Dict[str, Tuple[str, str, Tuple[int, int]]]:
        # Clear the cache for each new document
        self._get_dominant_font_size.cache_clear()
        
        if not os.path.exists(pdf_path):
            return {}
        extracted_paths = []
        try:
            with fitz.open(pdf_path) as doc:
                all_captions = self._locate_all_captions(doc)
                if not all_captions:
                    return {}

                logger.info(f"Found {len(all_captions)} captions. Starting font-aware engine.")

                # Build a map of page -> list of captions for quick lookup
                captions_by_page = {}
                for c in all_captions:
                    captions_by_page.setdefault(c.page_num, []).append(c)

                embedded_images_by_page = {}
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    try:
                        imgs = page.get_images(full=True)
                    except Exception:
                        imgs = []
                    img_entries = []
                    for img_info in imgs:
                        # get_image_rects may return multiple rects for the same xref
                        rects = [fitz.Rect(r) for r in page.get_image_rects(img_info)]
                        img_entries.append((img_info, rects))
                    if img_entries:
                        embedded_images_by_page[page_num] = img_entries

                # Decide per page whether embedded images match captions well enough
                extracted = []  # temporary list of tuples (caption, image_path, (w,h))

                for i, caption in enumerate(all_captions):
                    prev_caption = all_captions[i-1] if i > 0 else None
                    page = doc.load_page(caption.page_num)

                    image_saved = None

                    imgs_on_page = embedded_images_by_page.get(caption.page_num, [])
                    caps_on_page = captions_by_page.get(caption.page_num, [])

                    use_embedded_for_page = False
                    if imgs_on_page and len(imgs_on_page) >= len(caps_on_page):
                        # Heuristic: If there are at least as many images as captions, it's worth trying to match them.
                        use_embedded_for_page = True

                    if use_embedded_for_page:
                        # flatten rects so we can match spatially
                        rects = []
                        for img_info, rlist in imgs_on_page:
                            for r in rlist:
                                rects.append((img_info, r))
                        # prefer images that are above or near the caption and horizontally aligned
                        image_saved = self._try_extract_embedded_image_for_caption(page, caption, [r for _, r in rects])

                        if image_saved:
                            try:
                                # get pixel size from the saved file using cv2
                                img = cv2.imread(image_saved)
                                h, w = img.shape[:2] if img is not None else (0, 0)
                                extracted.append((caption, image_saved, (w, h)))
                                continue # Move to the next caption
                            except Exception:
                                # if something went wrong reading the saved image, fall back
                                pass

                    # Fallback to rendering-based extraction
                    image_saved = self._extract_figure_for_caption(page, caption, prev_caption, pdf_path)
                    if image_saved:
                        try:
                            img = cv2.imread(image_saved)
                            h, w = img.shape[:2] if img is not None else (0, 0)
                            extracted.append((caption, image_saved, (w, h)))
                        except Exception:
                            extracted.append((caption, image_saved, (0, 0)))

                # Build the final dict keyed by figure_x
                results = {}
                for idx, (cap, path, size) in enumerate(extracted, start=1):
                    key = f"figure_{idx}"
                    # cap is a FigureCaption object; return the full caption text string
                    results[key] = (cap.text, path, size)
            return results
        except Exception as e:
            logger.error(f"An error occurred during extraction: {e}", exc_info=True)
        return {}
    
    def _locate_all_captions(self, doc: fitz.Document) -> List[FigureCaption]:
        captions = []
        for page_num in range(len(doc)):
            for block in doc.load_page(page_num).get_text("blocks"):
                if block[6] == 0:
                    block_text = block[4].strip()
                    match = self.CAPTION_PATTERN.match(block_text)
                    if match:
                        label = f"{match.group(1).replace('.', '')}_{match.group(2)}"
                        captions.append(FigureCaption(page_num, label, fitz.Rect(block[:4]), block_text))
        captions.sort(key=lambda c: (c.page_num, c.rect.y0))
        return captions

    def _extract_figure_for_caption(self, page: fitz.Page, caption: FigureCaption, prev_cap: Optional[FigureCaption], pdf_path: str) -> Optional[str]:
        try:
            # Step 0: Get the page's dominant font size
            dominant_font_size = self._get_dominant_font_size(page)
            
            # Step 1 & 2: Infer layout and define a generous ROI
            search_roi = self._infer_layout_and_roi(page, caption, prev_cap)
            # Exclude document margins/content outside the main content area (use pdf_margin)
            try:
                margins = get_pdf_page_margins(pdf_path, page.number)
            except Exception:
                logger.warning(f"Failed to compute margins for page {page.number + 1}. Proceeding without margin exclusion.")
                margins = None

            if margins:
                # Construct content rect in PDF points
                page_rect = page.rect
                content_rect = fitz.Rect(
                    page_rect.x0 + margins.get('left', 0.0),
                    page_rect.y0 + margins.get('top', 0.0),
                    page_rect.x1 - margins.get('right', 0.0),
                    page_rect.y1 - margins.get('bottom', 0.0),
                )
                # Intersect search ROI with content rect to exclude marginal artifacts
                search_roi = search_roi & content_rect
                if search_roi.is_empty:
                    logger.info(f"Search ROI after excluding margins is empty for {caption.label}. Skipping.")
                    return None
            if search_roi.is_empty: return None

            pix = page.get_pixmap(matrix=fitz.Matrix(self.ZOOM_FACTOR, self.ZOOM_FACTOR))
            img_cv = self._pixmap_to_cv_image(pix)

            # Step 3, 4, 5: Use projection and font-aware merging to find the precise figure rect
            final_figure_rect = self._find_figure_by_projection(img_cv, search_roi, caption, dominant_font_size)
            if not final_figure_rect:
                logger.warning(f"Font-aware engine could not isolate a figure for {caption.label}.")
                return None
                
            return self._crop_and_save(page, final_figure_rect, caption)
        except Exception as e:
            logger.error(f"Fatal error processing {caption.label} on page {page.number + 1}: {e}", exc_info=True)
            return None

    def _infer_layout_and_roi(self, page: fitz.Page, caption: FigureCaption, prev_cap: Optional[FigureCaption]) -> fitz.Rect:
        page_rect, caption_rect = page.rect, caption.rect
        caption_center_x = (caption_rect.x0 + caption_rect.x1) / 2
        page_mid_x = page_rect.width / 2
        pad_x = 0

        # If caption is approximately centered, use full width. Otherwise restrict to the column half
        if abs(caption_center_x - page_mid_x) < page_rect.width * self.LAYOUT_CENTER_TOLERANCE:
            x0, x1 = 0, page_rect.width
        elif caption_center_x < page_mid_x:
            # left column: restrict to left half (with a small pad to avoid bleeding)
            x0, x1 = 0, max(0, page_mid_x - pad_x)
        else:
            # right column
            x0, x1 = min(page_mid_x + pad_x, page_rect.width), page_rect.width

        upper_bound_y = 0.0
        if prev_cap and prev_cap.page_num == page.number:
            upper_bound_y = max(upper_bound_y, prev_cap.rect.y0)
        y0, y1 = upper_bound_y, caption_rect.y0
        return (fitz.Rect(x0, y0, x1, y1) & page_rect)

    def _find_all_content_blocks(self, projection: np.ndarray) -> List[Tuple[int, int]]:
        content_threshold = np.max(projection) * self.PROJECTION_CONTENT_THRESHOLD_RATIO if np.max(projection) > 0 else 0
        in_content = False
        blocks, start_y = [], 0
        for y, value in enumerate(projection):
            if value > content_threshold and not in_content:
                in_content = True
                start_y = y
            elif value <= content_threshold and in_content:
                in_content = False
                if y - start_y > self.PROJECTION_WHITESPACE_THRESHOLD:
                    blocks.append((start_y, y))
        if in_content and len(projection) - start_y > self.PROJECTION_WHITESPACE_THRESHOLD:
            blocks.append((start_y, len(projection)))
        return blocks

    def _find_figure_by_projection(self, full_page_cv: np.ndarray, roi_pdf: fitz.Rect, caption: FigureCaption, font_size: float) -> Optional[fitz.Rect]:
        # Map PDF rect (points) to pixel coordinates consistently
        roi_pix = fitz.Rect(roi_pdf.x0 * self.ZOOM_FACTOR, roi_pdf.y0 * self.ZOOM_FACTOR,
        roi_pdf.x1 * self.ZOOM_FACTOR, roi_pdf.y1 * self.ZOOM_FACTOR)
        roi_coords = (max(0, int(round(roi_pix.y0))), min(full_page_cv.shape[0], int(round(roi_pix.y1))),
        max(0, int(round(roi_pix.x0))), min(full_page_cv.shape[1], int(round(roi_pix.x1))))
        roi_cv = full_page_cv[roi_coords[0]:roi_coords[1], roi_coords[2]:roi_coords[3]]
        
        if roi_cv.size == 0: return None
        
        gray = cv2.cvtColor(roi_cv, cv2.COLOR_BGR2GRAY)
        stddev = np.std(gray)
        mean_val = np.mean(gray)
        block_size = self.ADAPTIVE_THRESH_BLOCK_SIZE
        if block_size >= gray.shape[1]:
            block_size = max(3, gray.shape[1] // 3 * 2 + 1)

        if stddev < 8.0:
            if mean_val > 250:
                delta = 12
                thresh_val = int(max(1, mean_val - delta))
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            else:
                mean_img = np.full_like(gray, int(round(mean_val)))
                diff = cv2.absdiff(gray, mean_img)
                diff_thresh = int(max(6, stddev * 1.5))
                _, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        else:
            c_val = self.ADAPTIVE_THRESH_C
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, block_size, c_val)
        
        raw_projection = np.sum(thresh, axis=1)

        kernel_size = 13 
        if len(raw_projection) < kernel_size:
            kernel_size = len(raw_projection)
        projection = cv2.medianBlur(raw_projection.astype(np.uint8), kernel_size).flatten()

        content_blocks = self._find_all_content_blocks(projection)
        if not content_blocks: return None

        min_figure_height_px = self.MIN_FIGURE_HEIGHT_IN_FONTS * font_size * self.ZOOM_FACTOR * (1.2) # pt to px standard conversion
        final_y_start, final_y_end = content_blocks[-1]
        
        for i in range(len(content_blocks) - 2, -1, -1):
            current_height = final_y_end - final_y_start
            if current_height >= min_figure_height_px:
                break
            
            prev_block = content_blocks[i]
            final_y_start = prev_block[0]
            logger.debug(f"Block for {caption.label} too short ({current_height}px). Merging up.")

        if final_y_end - final_y_start < min_figure_height_px:
            logger.warning(f"Final merged block for {caption.label} is still too short. Result may be incorrect.")

        figure_zone_thresh = thresh[final_y_start:final_y_end, :]

        contours, _ = cv2.findContours(figure_zone_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)
        
        final_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        if self.debug:
            figure_zone_cv = roi_cv[final_y_start:final_y_end, :]
            self._save_debug_images(caption, roi_cv, projection, content_blocks, (final_y_start, final_y_end), figure_zone_cv, contours, final_bbox)
        
        final_x0 = roi_pix.x0 + final_bbox[0]
        final_y0 = roi_pix.y0 + final_y_start + final_bbox[1]
        final_x1 = final_x0 + final_bbox[2]
        final_y1 = final_y0 + final_bbox[3]
        
        return fitz.Rect(final_x0, final_y0, final_x1, final_y1) / self.ZOOM_FACTOR

    def _crop_and_save(self, page: fitz.Page, rect: fitz.Rect, caption: FigureCaption) -> Optional[str]:
        padded_rect = rect + (-3, 0, 3, 0) & page.rect
        if padded_rect.is_empty: return None
        pix = page.get_pixmap(matrix=fitz.Matrix(self.ZOOM_FACTOR, self.ZOOM_FACTOR), clip=padded_rect)
        safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "", caption.label)
        image_name = f"{safe_label.lower()}.png"
        image_path = os.path.join(self.output_dir, image_name)
        pix.save(image_path)
        logger.info(f"Successfully extracted: {image_path}")
        return image_path
        
    def _pixmap_to_cv_image(self, pix: fitz.Pixmap) -> np.ndarray:
        data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(data, cv2.COLOR_RGBA2BGR if pix.n==4 else cv2.COLOR_RGB2BGR)
    
    def _try_extract_embedded_image_for_caption(self, page: fitz.Page, caption: FigureCaption, img_rects: List[fitz.Rect]) -> Optional[str]:
        # Find images that lie above the caption and horizontally overlap the caption
        candidates = []
        for r in img_rects:
            # require the image to be above (or slightly overlapping) the caption vertically
            vertical_gap = caption.rect.y0 - r.y1
            if vertical_gap < -2.0:
                # image is below caption -> skip
                continue

            # compute overlaps
            inter = (r & caption.rect)
            # Use intersection over union (IoU) for horizontal overlap, more robust for columns
            union_x = max(r.x1, caption.rect.x1) - min(r.x0, caption.rect.x0)
            inter_x = max(0, min(r.x1, caption.rect.x1) - max(r.x0, caption.rect.x0))
            horizontal_iou = inter_x / union_x if union_x > 0 else 0

            # allow centered images that are near the caption even if overlap is small
            center_x_diff = abs((r.x0 + r.x1)/2 - (caption.rect.x0 + caption.rect.x1)/2)
            center_tol = max(10.0, page.rect.width * 0.1) # Tolerance relative to page width

            # A candidate is valid if it's reasonably close and horizontally aligned/overlapping
            if horizontal_iou > 0.1 or center_x_diff < center_tol:
                # Score based on vertical gap (closer is better) and horizontal alignment (centered is better)
                score = (vertical_gap, center_x_diff)
                candidates.append((r, score))

        if not candidates:
            return None

        # Prefer candidates closest vertically, with a tie-breaker for better horizontal centering
        candidates.sort(key=lambda tup: tup[1])
        best_rect = candidates[0][0]
        vertical_distance = candidates[0][1][0]

        # --- VALIDATION STAGE ---
        
        # 1. Reject if best candidate is still too far vertically
        if vertical_distance > max(50.0, caption.rect.height * 4):
            logger.info(f"Rejecting embedded candidate for {caption.label}: too far vertically ({vertical_distance:.1f} pts).")
            return None

        # 2. **NEW**: Reject if the image dimensions are too small (likely an icon, not a figure)
        if (best_rect.width < self.MIN_EMBEDDED_IMG_ABS_DIM and 
            best_rect.height < self.MIN_EMBEDDED_IMG_ABS_DIM):
            logger.info(f"Rejecting embedded candidate for {caption.label}: dimensions ({best_rect.width:.1f}x{best_rect.height:.1f} pts) are too small.")
            return None
        
        # 3. **NEW**: Reject if the image is significantly narrower than its caption text
        if best_rect.width < caption.rect.width * self.MIN_EMBEDDED_IMG_WIDTH_RATIO_VS_CAPTION:
            logger.info(f"Rejecting embedded candidate for {caption.label}: too narrow ({best_rect.width:.1f} pts) compared to caption width ({caption.rect.width:.1f} pts).")
            return None

        # If all checks pass, extract the image
        try:
            # Use a small padding to ensure edges are not cut off, but clip to page boundaries
            padded_rect = best_rect + (-2, -2, 2, 2)
            padded_rect.intersect(page.rect)
            if padded_rect.is_empty: return None

            full_pix = page.get_pixmap(matrix=fitz.Matrix(self.ZOOM_FACTOR, self.ZOOM_FACTOR), clip=padded_rect)
            
            # Additional check: if the pixmap is tiny, it might be an error
            if full_pix.width < 10 or full_pix.height < 10:
                 return None

            image_bytes = full_pix.tobytes(output="png")
            safe_label = re.sub(r"[^a-zA-Z0--9_\-]", "", caption.label)
            image_name = f"{safe_label.lower()}.png"
            image_path = os.path.join(self.output_dir, image_name)
            with open(image_path, "wb") as fh:
                fh.write(image_bytes)
            logger.info(f"Extracted embedded image for {caption.label} after passing validation: {image_path}")
            return image_path
        except Exception as e:
            logger.warning(f"Failed to save validated embedded image for {caption.label}: {e}")
            return None
    
    def _save_debug_images(self, cap, roi_cv, projection, all_blocks, final_zone_y, zone_cv, all_contours, bbox):
        prefix = f"p{cap.page_num + 1}_{cap.label}"
        cv2.imwrite(os.path.join(self.debug_dir, f"{prefix}_01_roi.png"), roi_cv)
        if plt:
            plt.figure(figsize=(10, 5)); plt.plot(projection, range(len(projection))); plt.ylim(len(projection), 0)
            for i, (start, end) in enumerate(all_blocks):
                plt.axhline(y=start, color='c', linestyle=':', linewidth=1)
                plt.axhline(y=end, color='c', linestyle=':', linewidth=1)
                plt.text(np.max(projection) * 1.05, start + (end - start) / 2, f"Block {i}", verticalalignment='center')
            plt.axhspan(final_zone_y[0], final_zone_y[1], color='g', alpha=0.3, label='Final Figure Zone')
            plt.title('Horizontal Projection & Block Analysis'); plt.legend(); plt.savefig(os.path.join(self.debug_dir, f"{prefix}_02_projection.png")); plt.close()
        cv2.imwrite(os.path.join(self.debug_dir, f"{prefix}_03_figure_zone.png"), zone_cv)
        zone_contours_img = zone_cv.copy()
        cv2.drawContours(zone_contours_img, all_contours, -1, (255, 0, 0), 1)
        x,y,w,h = bbox
        cv2.rectangle(zone_contours_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.debug_dir, f"{prefix}_04_contours.png"), zone_contours_img)

class PyMuPDFExtractor(PyMuPDFFontAwareEngine):
    """Compatibility wrapper with the older name used across the project/tests."""
    def __init__(self, output_dir: str = "data", debug: bool = False):
        super().__init__(output_dir=output_dir, debug=debug)

    def extract_images(self, pdf_path: str):
        return super().extract_images(pdf_path)

if __name__ == "__main__":
    test_pdf_path = "./data/2305.18323/2305.18323.pdf" 
    extractor = PyMuPDFExtractor(output_dir="./data/debug", debug=True)
    if os.path.exists(test_pdf_path):
        results = extractor.extract_images(test_pdf_path)
        print(f"\n--- Results for {os.path.basename(test_pdf_path)} ---")
        print(f"Total images extracted: {len(results)}")
        for k, (cap_text, path, size) in results.items():
            print(f"{k}: caption={cap_text.strip()[:60]}..., path={path}, size={size}")
    else:
        print(f"Test PDF not found at {test_pdf_path}.")
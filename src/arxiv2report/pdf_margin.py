import fitz  # PyMuPDF
from typing import Dict, Union, List, Tuple

def get_pdf_page_margins(
    pdf_path: str,
    page_number: int,
) -> Union[Dict[str, float], None]:
    """
    计算给定PDF文件指定页面的内容边界，从而推导出页边距，

    该函数通过联合页面上所有文本、图像和绘图元素的边界框来确定
    整体内容的边界框。特别地，它包含了针对arXiv PDF第一页左侧
    垂直水印的排除逻辑。

    Args:
        pdf_path (str): PDF文件的路径。
        page_number (int): 要处理的页面编号 (从0开始)。

    Returns:
        Union[Dict[str, float], None]: 一个包含'top', 'bottom', 'left', 'right'
                                        页边距值的字典 (单位: points)。
                                        如果页面为空或无法处理，则返回None。
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return None

    if not (0 <= page_number < len(doc)):
        print(f"Error: Page number {page_number} is out of range "
              f"for a document with {len(doc)} pages.")
        doc.close()
        return None

    page = doc[page_number]
    page_rect = page.rect
    # Start with an invalid rect and expand it with include_rect when content is found.
    content_bbox = fitz.Rect()  # empty/invalid rect

    # 1. 获取所有文本块并进行预处理
    text_blocks: List[Tuple] = page.get_text("blocks")

    # 针对arXiv首页的特殊处理逻辑：更强健地识别并排除竖向水印
    excluded_rects: List[fitz.Rect] = []
    if page_number == 0:
        filtered_blocks = []
        for block in text_blocks:
            rect = fitz.Rect(block[:4])
            text = (block[4] or "").strip()

            # 启发式规则识别arXiv的竖向水印：
            # - 文本中包含 'arxiv'（不区分大小写），或
            # - 矩形位于页面左侧很窄但很高（可能是图形化的水印）
            near_left_edge = rect.x1 < page.rect.width * 0.15
            tall_narrow = rect.height > rect.width * 2.5
            contains_arxiv = 'arxiv' in text.lower()

            is_arxiv_watermark = (contains_arxiv and near_left_edge) or (near_left_edge and tall_narrow)

            if not is_arxiv_watermark:
                filtered_blocks.append(block)
            else:
                excluded_rects.append(rect & page.rect)
                print("Info: Potential arXiv watermark detected and excluded on page 0.")

        text_blocks = filtered_blocks  # 使用过滤后的文本块列表

    # 将有效文本块的边界框合并到总内容边界框中
    for block in text_blocks:
        try:
            rect = fitz.Rect(block[:4])
            if not content_bbox.is_valid:
                content_bbox = fitz.Rect(rect)
            else:
                content_bbox |= rect
        except Exception:
            # skip malformed blocks
            continue

    # 2. 获取所有图像的边界框并合并
    images = page.get_images(full=True)
    for img_info in images:
        img_rects = page.get_image_rects(img_info)
        for rect in img_rects:
            try:
                r = fitz.Rect(rect)
                # 如果图像矩形与已识别的被排除区域高度重叠，则跳过
                skip = False
                for ex in excluded_rects:
                    inter = (r & ex)
                    if inter.is_valid and not inter.is_empty:
                        # overlap ratio > 50% => likely watermark graphic
                        overlap_area = inter.get_area()
                        if overlap_area / max(1.0, r.get_area()) > 0.5:
                            skip = True
                            break
                if skip:
                    continue
                if not content_bbox.is_valid:
                    content_bbox = fitz.Rect(r)
                else:
                    content_bbox |= r
            except Exception:
                continue

    # 3. 获取所有矢量绘图的边界框并合并
    drawings = page.get_drawings()
    for path in drawings:
        try:
            r = fitz.Rect(path.get("rect", None))
            # 跳过与被排除区域重叠较多的绘图元素
            skip = False
            for ex in excluded_rects:
                inter = (r & ex)
                if inter.is_valid and not inter.is_empty:
                    if inter.get_area() / max(1.0, r.get_area()) > 0.5:
                        skip = True
                        break
            if skip:
                continue
            if not content_bbox.is_valid:
                content_bbox = fitz.Rect(r)
            else:
                content_bbox |= r
        except Exception:
            continue

    # 如果页面没有可识别的内容，content_bbox将是无效或空的
    # Ensure content_bbox is a valid rectangle inside the page. If not, fall back to full page.
    if not content_bbox.is_valid or content_bbox.is_empty:
        print(f"Warning: No valid content found on page {page_number} to determine margins.")
        # 将整个页面渲染为图像以供检查
        pix = page.get_pixmap()
        doc.close()
        return {
            "left": 0.0,
            "top": 0.0,
            "right": page.width,
            "bottom": page.height
        }

    # 计算页边距
    # Clamp content_bbox to page rect to avoid negative margins due to out-of-page content coords
    content_bbox = content_bbox & page_rect

    margins = {
        "left": max(0.0, content_bbox.x0 - page_rect.x0),
        "top": max(0.0, content_bbox.y0 - page_rect.y0),
        "right": max(0.0, page_rect.x1 - content_bbox.x1),
        "bottom": max(0.0, page_rect.y1 - content_bbox.y1),
    }

    doc.close()
    return margins


if __name__ == '__main__':
    # --- 使用示例 ---
    PDF_FILE = "data/1.pdf"
    PAGE_TO_ANALYZE = 0  # 分析第一页 (索引从0开始)
    OUTPUT_IMAGE = "cropped_content.png"

    try:
        # 调用函数
        calculated_margins = get_pdf_page_margins_and_crop(
            pdf_path=PDF_FILE,
            page_number=PAGE_TO_ANALYZE,
            output_image_path=OUTPUT_IMAGE
        )

        # 打印结果
        if calculated_margins is not None:
            print(f"Analysis of '{PDF_FILE}', page {PAGE_TO_ANALYZE + 1}:")
            for margin, value in calculated_margins.items():
                print(f"  - {margin.capitalize():<7}: {value:.2f} points")
            print(f"\nCropped content area has been saved to '{OUTPUT_IMAGE}' for verification.")

    except FileNotFoundError:
        print(f"Error: The file '{PDF_FILE}' was not found.")
        print("Please ensure the PDF file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
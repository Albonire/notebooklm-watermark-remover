#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fitz  # PyMuPDF
import argparse
import os
import logging
import cv2
import numpy as np
import zipfile
import shutil
import tempfile
from typing import Optional, List
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import io

# ------------------------------------------------------------------ #
#  Logging configuration
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
@dataclass
class WatermarkConfig:
    """Configuration for watermark detection and removal."""
    # Search margins from the bottom-right corner (PDF pts / image px)
    search_margin_x: int = 300
    search_margin_y: int = 65

    # Padding around the tight watermark bbox before inpainting (pixels)
    watermark_padding: int = 8

    # ---- New parameters for bilateral background estimation ----
    # Bilateral filter diameter (larger values smooth more)
    bilateral_d: int = 15
    # Sigma for color similarity (higher = more blur on different colors)
    bilateral_sigma_color: int = 75
    # Sigma for spatial distance (higher = more blur)
    bilateral_sigma_space: int = 75

    # Threshold for difference image (grayscale, 0-255)
    pixel_threshold: int = 20

    # Inpainting radius for cv2.inpaint
    inpaint_radius: int = 5

    # Morphological kernel size for closing gaps in the binary mask
    morph_close_kernel: int = 3

    # Minimum total pixel area of selected watermark components
    min_watermark_area: int = 500

    # Per-component area threshold (filters tiny noise)
    min_component_area: int = 100

    # PDF rendering scale factor (higher = better quality but slower)
    pdf_dpi_scale: float = 3.0


# ------------------------------------------------------------------ #
#  Core engine
# ------------------------------------------------------------------ #
class WatermarkRemover:
    """Removes NotebookLM watermarks from PDFs, images, and PPTX files."""

    WATERMARK_TEXT = "NotebookLM"

    def __init__(self, config: WatermarkConfig = WatermarkConfig()):
        self.config = config

    # ---------- detection helpers ---------- #
    def _build_watermark_mask(
        self, roi_bgr: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Build a pixel-precise inpainting mask that covers ONLY the watermark
        component pixels inside *roi_bgr*.

        Uses a bilateral filter to estimate background, then thresholds the
        difference. Component filtering is relaxed because the ROI is already
        the watermark region.
        """
        h, w = roi_bgr.shape[:2]
        if h < 5 or w < 5:
            return None

        # 1. Estimate background with edge-preserving bilateral filter.
        #    The filter blurs fine details like text while keeping strong edges.
        bg = cv2.bilateralFilter(
            roi_bgr,
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space,
        )

        # 2. Compute difference in grayscale (or use LAB lightness)
        gray_orig = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_orig, gray_bg)

        # 3. Threshold to get candidate watermark pixels
        _, binary = cv2.threshold(
            diff, self.config.pixel_threshold, 255, cv2.THRESH_BINARY
        )

        # 4. Close small gaps (characters often break into many parts)
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_close_kernel, self.config.morph_close_kernel),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 5. Connected components analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 6. Select components that are large enough and not spanning the whole ROI
        wm_labels: List[int] = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            # Ignore tiny noise
            if area < self.config.min_component_area:
                continue
            # Ignore components that are extremely wide/tall (full-span borders, etc.)
            if cw > w * 0.9 or ch > h * 0.9:
                continue
            wm_labels.append(i)

        if len(wm_labels) == 0:
            return None

        # 7. Check total area (solid watermark area after closing)
        total_area = sum(int(stats[i][4]) for i in wm_labels)
        if total_area < self.config.min_watermark_area:
            return None

        # 8. Build clean mask (only watermark pixels)
        mask = np.zeros((h, w), dtype=np.uint8)
        for lid in wm_labels:
            mask[labels == lid] = 255

        # 9. Dilate slightly to cover anti-aliased/soft edges around glyphs
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=2)

        return mask if cv2.countNonZero(mask) > 0 else None

    def _has_watermark(self, roi_bgr: np.ndarray) -> bool:
        """True if the ROI contains a watermark."""
        return self._build_watermark_mask(roi_bgr) is not None

    # ---------- inpainting ---------- #
    def _inpaint_region(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Inpaints ALL detected sharp features in *img_bgr*.
        This should only be called on a tightly-cropped region where
        everything that differs from the background IS watermark.
        """
        try:
            h, w = img_bgr.shape[:2]
            if h < 5 or w < 5:
                return img_bgr

            # We also use bilateral filter here for consistency,
            # but it's essentially the same logic as _clean_watermark_in_roi
            # without component filtering (since we know it's all watermark).
            bg = cv2.bilateralFilter(
                img_bgr,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space,
            )
            diff_gray = cv2.cvtColor(cv2.absdiff(img_bgr, bg), cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                diff_gray, self.config.pixel_threshold, 255, cv2.THRESH_BINARY
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)

            if cv2.countNonZero(mask) == 0:
                return img_bgr

            return cv2.inpaint(
                img_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA
            )
        except Exception as e:
            logger.warning(f"Inpainting failed: {e}")
            return img_bgr

    def _clean_watermark_in_roi(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Precision cleaning: builds a pixel-precise mask covering only the
        watermark components, then inpaints those specific pixels.
        Surrounding content (borders, decorations, slide text) is untouched.
        Returns the modified ROI, or None if no watermark is detected.
        """
        mask = self._build_watermark_mask(roi_bgr)
        if mask is None:
            return None

        return cv2.inpaint(
            roi_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA
        )

    # ------------------------------------------------------------------ #
    #  PDF processing
    # ------------------------------------------------------------------ #
    def _find_watermark_rect_text(self, page: fitz.Page) -> Optional[fitz.Rect]:
        """Locate watermark via PDF text search.  Returns padded Rect or None."""
        w, h = page.rect.width, page.rect.height
        instances = page.search_for(self.WATERMARK_TEXT)
        if not instances:
            return None

        best: Optional[fitz.Rect] = None
        best_score = float('inf')

        for rect in instances:
            cy = (rect.y0 + rect.y1) / 2
            cx = (rect.x0 + rect.x1) / 2
            if cy < h * 0.80 or rect.width > 250 or rect.height > 40:
                continue
            dist = abs(w - cx) + abs(h - cy)
            if dist < best_score:
                best_score = dist
                best = rect

        if best is None:
            return None

        wm_rect = fitz.Rect(best)

        # Expand left to catch the icon
        icon_zone = fitz.Rect(best.x0 - 80, best.y0 - 15, best.x0 + 5, best.y1 + 15)
        try:
            for d in page.get_drawings():
                if d["rect"].intersects(icon_zone):
                    wm_rect = wm_rect | d["rect"]
        except Exception:
            pass
        try:
            for img_info in page.get_images(full=True):
                for ir in page.get_image_rects(img_info[0]):
                    if ir.intersects(icon_zone):
                        wm_rect = wm_rect | ir
        except Exception:
            pass

        wm_rect.x0 = min(wm_rect.x0, best.x0 - 45)
        pad = self.config.watermark_padding
        return fitz.Rect(
            max(0, wm_rect.x0 - pad),
            max(0, wm_rect.y0 - pad),
            min(w, wm_rect.x1 + pad),
            min(h, wm_rect.y1 + pad),
        )

    def _pixmap_to_bgr(self, pix: fitz.Pixmap) -> Optional[np.ndarray]:
        """Convert a PyMuPDF Pixmap to a BGR numpy array."""
        data = np.frombuffer(pix.samples, dtype=np.uint8)
        if pix.n == 4:
            return cv2.cvtColor(data.reshape(pix.h, pix.w, 4), cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(data.reshape(pix.h, pix.w, 3), cv2.COLOR_RGB2BGR)
        return None

    def _patch_pdf_rect(self, page: fitz.Page, rect: fitz.Rect, precision: bool = True) -> bool:
        """
        Rasterise *rect*, clean watermark, paste back.
        If *precision* is True, only the tight sub-region is inpainted.
        """
        mat = fitz.Matrix(self.config.pdf_dpi_scale, self.config.pdf_dpi_scale)
        pix = page.get_pixmap(clip=rect, matrix=mat)
        roi_bgr = self._pixmap_to_bgr(pix)
        if roi_bgr is None:
            return False

        if precision:
            cleaned = self._clean_watermark_in_roi(roi_bgr)
            if cleaned is None:
                return False
        else:
            cleaned = self._inpaint_region(roi_bgr)

        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(cleaned_rgb).save(buf, format='PNG')
        page.insert_image(rect, stream=buf.getvalue())
        return True

    def process_pdf(self, input_path: str, output_path: str, preview: bool = False) -> bool:
        """Process a PDF: detect watermark per page and inpaint it."""
        try:
            doc = fitz.open(input_path)
        except Exception as e:
            logger.error(f"Could not open {input_path}: {e}")
            return False

        filename = os.path.basename(input_path)
        pbar = tqdm(enumerate(doc), total=len(doc), desc=f"Processing {filename}", unit="page")
        patched = skipped = 0

        for i, page in pbar:
            if preview and i > 0:
                break

            w, h = page.rect.width, page.rect.height

            # Strategy 1: text-based (vector PDFs)
            wm_rect = self._find_watermark_rect_text(page)
            if wm_rect is not None:
                if self._patch_pdf_rect(page, wm_rect, precision=False):
                    patched += 1
                    pbar.set_postfix(patched=patched, skipped=skipped)
                    continue

            # Strategy 2: pixel-based (raster PDFs — common for NotebookLM)
            corner = fitz.Rect(max(0, w - self.config.search_margin_x),
                               max(0, h - self.config.search_margin_y), w, h)
            if self._patch_pdf_rect(page, corner, precision=True):
                patched += 1
            else:
                skipped += 1
            pbar.set_postfix(patched=patched, skipped=skipped)

        try:
            doc.save(output_path, garbage=3, deflate=True, clean=True)
            doc.close()
            logger.info(f"Saved {output_path} ({patched} patched, {skipped} skipped)")
            return True
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Image processing
    # ------------------------------------------------------------------ #
    def _clean_roi_scaled(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Upscale *roi_bgr* by ``pdf_dpi_scale``, run watermark detection and
        inpainting at the higher resolution, then downscale the result back.
        """
        scale = self.config.pdf_dpi_scale
        h, w = roi_bgr.shape[:2]
        roi_hr = cv2.resize(roi_bgr, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR)
        cleaned_hr = self._clean_watermark_in_roi(roi_hr)
        if cleaned_hr is None:
            return None
        return cv2.resize(cleaned_hr, (w, h),
                          interpolation=cv2.INTER_LINEAR)

    def process_image(self, input_path: str, output_path: str) -> bool:
        """Remove watermark from a standalone image file (PNG/JPG/WEBP)."""
        try:
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Could not read: {input_path}")
                return False

            h, w = img.shape[:2]
            has_alpha = len(img.shape) == 3 and img.shape[2] == 4

            if has_alpha:
                channels = cv2.split(img)
                img_bgr = cv2.merge(channels[:3])
                alpha = channels[3]
            else:
                img_bgr = img.copy()
                alpha = None

            mx, my = self.config.search_margin_x, self.config.search_margin_y
            y0 = max(0, h - my)
            x0 = max(0, w - mx)

            roi = img_bgr[y0:h, x0:w].copy()
            cleaned_roi = self._clean_roi_scaled(roi)

            if cleaned_roi is None:
                logger.warning(f"No watermark detected in {input_path}")
                return False

            img_bgr[y0:h, x0:w] = cleaned_roi

            if has_alpha:
                img_final = cv2.merge([*cv2.split(img_bgr), alpha])
            else:
                img_final = img_bgr

            cv2.imwrite(output_path, img_final)
            logger.info(f"Saved cleaned image to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  PPTX processing
    # ------------------------------------------------------------------ #
    def _clean_pptx_image_bytes(self, img_bytes: bytes) -> Optional[bytes]:
        """
        Decodes an image from raw bytes, removes the watermark from the
        bottom-right corner, and re-encodes.  Returns None on failure or
        when no watermark is detected.
        """
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        h, w = img.shape[:2]
        has_alpha = len(img.shape) == 3 and img.shape[2] == 4

        if has_alpha:
            channels = cv2.split(img)
            img_bgr = cv2.merge(channels[:3])
            alpha = channels[3]
        else:
            img_bgr = img.copy()
            alpha = None

        mx, my = self.config.search_margin_x, self.config.search_margin_y
        y0, x0 = max(0, h - my), max(0, w - mx)

        roi = img_bgr[y0:h, x0:w].copy()
        cleaned_roi = self._clean_roi_scaled(roi)
        if cleaned_roi is None:
            return None

        img_bgr[y0:h, x0:w] = cleaned_roi

        if has_alpha:
            img_final = cv2.merge([*cv2.split(img_bgr), alpha])
        else:
            img_final = img_bgr

        ok, encoded = cv2.imencode('.png', img_final)
        return encoded.tobytes() if ok else None

    def process_pptx(self, input_path: str, output_path: str) -> bool:
        """
        Processes a PPTX file by opening it as a ZIP archive, finding every
        embedded slide image, removing the watermark via inpainting, and
        saving the modified PPTX.
        """
        try:
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(input_path, 'r') as zin:
                zin.extractall(tmpdir)

            media_dir = os.path.join(tmpdir, 'ppt', 'media')
            if not os.path.isdir(media_dir):
                logger.error(f"No media directory in {input_path}")
                shutil.rmtree(tmpdir)
                return False

            image_exts = ('.png', '.jpg', '.jpeg', '.webp')
            images = sorted([
                f for f in os.listdir(media_dir)
                if f.lower().endswith(image_exts)
            ])

            if not images:
                logger.error(f"No images found in {input_path}")
                shutil.rmtree(tmpdir)
                return False

            patched = 0
            pbar = tqdm(images, desc=f"Processing {os.path.basename(input_path)}", unit="img")

            for img_name in pbar:
                img_path = os.path.join(media_dir, img_name)
                with open(img_path, 'rb') as f:
                    original = f.read()

                cleaned = self._clean_pptx_image_bytes(original)
                if cleaned is not None:
                    with open(img_path, 'wb') as f:
                        f.write(cleaned)
                    patched += 1
                pbar.set_postfix(patched=patched)

            # Re-pack the ZIP
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        arcname = os.path.relpath(full_path, tmpdir)
                        zout.write(full_path, arcname)

            shutil.rmtree(tmpdir)
            logger.info(f"Saved {output_path} ({patched}/{len(images)} images patched)")
            return True

        except Exception as e:
            logger.error(f"Error processing PPTX {input_path}: {e}")
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
            return False


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="NotebookLM Watermark Remover — PDF, Images & PPTX"
    )
    parser.add_argument("path", help="File (PDF/PPTX/PNG/JPG) or directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument(
        "--preview", action="store_true",
        help="Process only first page (PDF only)",
    )
    parser.add_argument(
        "--margin-x", type=int, default=None,
        help="Search margin width in px from right edge (default: 300).",
    )
    parser.add_argument(
        "--margin-y", type=int, default=None,
        help="Search margin height in px from bottom edge (default: 65).",
    )
    parser.add_argument(
        "--bilateral-d", type=int, default=None,
        help="Bilateral filter diameter (default: 15).",
    )
    parser.add_argument(
        "--bilateral-sigma-color", type=int, default=None,
        help="Bilateral filter sigma color (default: 75).",
    )
    parser.add_argument(
        "--bilateral-sigma-space", type=int, default=None,
        help="Bilateral filter sigma space (default: 75).",
    )
    parser.add_argument(
        "--pixel-threshold", type=int, default=None,
        help="Threshold for watermark difference (default: 20).",
    )

    args = parser.parse_args()
    config = WatermarkConfig()

    if args.margin_x is not None:
        config.search_margin_x = args.margin_x
    if args.margin_y is not None:
        config.search_margin_y = args.margin_y
    if args.bilateral_d is not None:
        config.bilateral_d = args.bilateral_d
    if args.bilateral_sigma_color is not None:
        config.bilateral_sigma_color = args.bilateral_sigma_color
    if args.bilateral_sigma_space is not None:
        config.bilateral_sigma_space = args.bilateral_sigma_space
    if args.pixel_threshold is not None:
        config.pixel_threshold = args.pixel_threshold

    remover = WatermarkRemover(config)

    supported = ('.pdf', '.pptx', '.png', '.jpg', '.jpeg', '.webp')

    if os.path.isdir(args.path):
        tasks = sorted([
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if f.lower().endswith(supported)
        ])
        logger.info(f"Found {len(tasks)} supported files.")
    elif os.path.isfile(args.path) and args.path.lower().endswith(supported):
        tasks = [args.path]
    else:
        logger.error("Invalid path or unsupported format.")
        return

    for input_path in tasks:
        ext = os.path.splitext(input_path)[1].lower()

        if args.output and len(tasks) == 1:
            out_path = args.output
        else:
            base, _ = os.path.splitext(input_path)
            out_path = f"{base}_cleaned{ext}"

        if ext == '.pdf':
            remover.process_pdf(input_path, out_path, preview=args.preview)
        elif ext == '.pptx':
            remover.process_pptx(input_path, out_path)
        else:
            remover.process_image(input_path, out_path)


if __name__ == "__main__":
    main()
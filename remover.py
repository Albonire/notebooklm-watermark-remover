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
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #

@dataclass
class WatermarkConfig:
    """Configuration for watermark detection and removal."""
    # Search margins from the bottom-right corner
    search_margin_x: int = 400
    search_margin_y: int = 120

    # Extra padding around detected watermark bbox
    watermark_padding: int = 6

    # Threshold for contrast-based candidate extraction
    pixel_threshold: int = 22

    # PDF rendering scale factor (higher = better quality)
    pdf_dpi_scale: float = 3.5

    # Inpainting radius for cv2.inpaint (used only as fallback)
    inpaint_radius: int = 3

    # Component filters
    min_watermark_components: int = 1
    min_watermark_area: int = 400
    min_component_area: int = 18
    max_component_area_ratio: float = 0.25

    # Text/template detection
    text_match_threshold: float = 0.45
    dark_text_threshold: int = 210
    roi_bottom_bias: float = 0.35
    roi_right_bias: float = 0.45

    # Morphology
    dilate_iterations: int = 1
    close_iterations: int = 1

    # Reconstruction
    use_patch_heal: bool = True
    patch_offset_x: int = -80  # Look 80px to the left for clean background
    patch_offset_y: int = -80  # Or 80px above

    # Debug
    debug: bool = False


# ------------------------------------------------------------------ #
#  Core engine                                                       #
# ------------------------------------------------------------------ #

class WatermarkRemover:
    """Removes a bottom-right NotebookLM-style watermark using patch-based reconstruction."""

    WATERMARK_TEXT = "NotebookLM"

    def __init__(self, config: WatermarkConfig = WatermarkConfig()):
        self.config = config
        self._template_cache = {}

    # ---------- utils ---------- #

    def _debug_save(self, name: str, img: np.ndarray) -> None:
        if not self.config.debug:
            return
        try:
            os.makedirs("debug_watermark", exist_ok=True)
            cv2.imwrite(os.path.join("debug_watermark", name), img)
        except Exception:
            pass

    def _pixmap_to_bgr(self, pix: fitz.Pixmap) -> Optional[np.ndarray]:
        data = np.frombuffer(pix.samples, dtype=np.uint8)
        if pix.n == 4:
            return cv2.cvtColor(data.reshape(pix.h, pix.w, 4), cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(data.reshape(pix.h, pix.w, 3), cv2.COLOR_RGB2BGR)
        if pix.n == 1:
            gray = data.reshape(pix.h, pix.w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return None

    def _render_text_template(self, height: int) -> np.ndarray:
        """Creates a binary template for 'NotebookLM' at a target height."""
        key = max(10, int(height))
        if key in self._template_cache:
            return self._template_cache[key]

        font_size = max(12, int(key * 1.15))
        canvas_w = max(180, font_size * 14)
        canvas_h = max(40, font_size * 3)

        img = Image.new('L', (canvas_w, canvas_h), 255)
        draw = ImageDraw.Draw(img)

        font = None
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), self.WATERMARK_TEXT, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = 8
        y = max(4, (canvas_h - th) // 2 - bbox[1])
        draw.text((x, y), self.WATERMARK_TEXT, fill=0, font=font)

        arr = np.array(img)
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0 or len(ys) == 0:
            tpl = np.zeros((10, 80), dtype=np.uint8)
        else:
            tpl = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        self._template_cache[key] = tpl
        return tpl

    def _template_match_text(self, roi_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Template-match the watermark text in the bottom-right ROI."""
        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 80:
            return None, 0.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        best_score = 0.0
        best_box = None

        # Restrict to dark-on-light candidates
        gray_eq = cv2.equalizeHist(gray)

        for text_h in range(max(14, h // 5), max(18, min(h - 2, h // 2 + 20)), 3):
            tpl = self._render_text_template(text_h)
            th, tw = tpl.shape[:2]
            if th >= h or tw >= w:
                continue

            result = cv2.matchTemplate(gray_eq, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                x, y = max_loc
                best_score = float(max_val)
                best_box = (x, y, tw, th)

        if best_score < self.config.text_match_threshold:
            return None, best_score
        return best_box, best_score

    def _extract_dark_candidates(self, roi_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Robust background estimate
        ksize = max(15, min(41, ((min(gray.shape[:2]) // 5) | 1)))
        bg = cv2.medianBlur(gray, ksize)
        diff_dark = cv2.subtract(bg, gray)

        # Dark pixels on light background
        dark_mask = np.where(gray < self.config.dark_text_threshold, 255, 0).astype(np.uint8)
        _, diff_mask = cv2.threshold(diff_dark, self.config.pixel_threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(dark_mask, diff_mask)

        # Restrict to bottom-right biased region to reduce false positives
        h, w = gray.shape[:2]
        geom = np.zeros_like(mask)
        x0 = int(w * self.config.roi_right_bias)
        y0 = int(h * self.config.roi_bottom_bias)
        geom[y0:h, x0:w] = 255
        mask = cv2.bitwise_and(mask, geom)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)
        mask = cv2.dilate(mask, kernel, iterations=self.config.dilate_iterations)
        return mask

    def _component_boxes_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = mask.shape[:2]
        out = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < self.config.min_component_area:
                continue
            if area > int(h * w * self.config.max_component_area_ratio):
                continue
            out.append((int(x), int(y), int(cw), int(ch), int(area)))
        return out

    def _find_icon_like_component(self, comps: List[Tuple[int, int, int, int, int]], text_box: Tuple[int, int, int, int], roi_shape) -> Optional[Tuple[int, int, int, int]]:
        tx, ty, tw, th = text_box
        best = None
        best_score = -1e9
        h, w = roi_shape[:2]

        for x, y, cw, ch, area in comps:
            # Expected icon to be left of text and near same vertical band
            cx = x + cw / 2
            cy = y + ch / 2
            text_cy = ty + th / 2
            if cx >= tx:
                continue
            if abs(cy - text_cy) > max(18, th * 0.9):
                continue
            if cw > th * 1.8 or ch > th * 1.8:
                continue
            if x < w * 0.45:
                continue
            score = -abs((tx - (x + cw)) - max(4, th * 0.15)) - abs(ch - th * 0.65) + area * 0.02
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)
        return best

    def _build_watermark_mask(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Hybrid watermark detection:
        1) detect dark components in the bottom-right region,
        2) locate text using template matching,
        3) fuse text-like components and optional icon,
        4) return a tight but safe mask for removal.
        """
        h, w = roi_bgr.shape[:2]
        if h < 10 or w < 20:
            return None

        candidate_mask = self._extract_dark_candidates(roi_bgr)
        comps = self._component_boxes_from_mask(candidate_mask)
        if not comps:
            return None

        text_box, score = self._template_match_text(roi_bgr)
        if text_box is None:
            # fallback: try union of bottom-right compact components
            selected = []
            for x, y, cw, ch, area in comps:
                cx = x + cw / 2
                cy = y + ch / 2
                if cx < w * 0.60 or cy < h * 0.55:
                    continue
                if ch > h * 0.7 or cw > w * 0.8:
                    continue
                selected.append((x, y, cw, ch, area))
            if not selected:
                return None

            mask = np.zeros((h, w), dtype=np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
            for i in range(1, n):
                x, y, cw, ch, area = stats[i]
                for sx, sy, sw, sh, sa in selected:
                    if x == sx and y == sy and cw == sw and ch == sh and area == sa:
                        mask[labels == i] = 255
                        break

            total_area = cv2.countNonZero(mask)
            if total_area < self.config.min_watermark_area:
                return None

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=2)
            self._debug_save("fallback_mask.png", mask)
            return mask

        tx, ty, tw, th = text_box
        pad = self.config.watermark_padding
        text_rect = (
            max(0, tx - pad),
            max(0, ty - pad),
            min(w, tx + tw + pad),
            min(h, ty + th + pad),
        )

        selected_mask = np.zeros((h, w), dtype=np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

        # Keep components overlapping/near text rect
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < self.config.min_component_area:
                continue
            rx0, ry0, rx1, ry1 = text_rect
            overlaps = not (x + cw < rx0 or x > rx1 or y + ch < ry0 or y > ry1)
            near = (x <= rx1 + 14 and x + cw >= rx0 - 18 and y <= ry1 + 10 and y + ch >= ry0 - 10)
            if overlaps or near:
                selected_mask[labels == i] = 255

        # Optional icon left of text
        icon_box = self._find_icon_like_component(comps, text_box, roi_bgr.shape)
        if icon_box is not None:
            ix, iy, iw, ih = icon_box
            for i in range(1, n):
                x, y, cw, ch, area = stats[i]
                if x == ix and y == iy and cw == iw and ch == ih:
                    selected_mask[labels == i] = 255

        # If template says text exists but selected mask is too thin, force text band from dark pixels
        text_band = candidate_mask[max(0, ty - 3):min(h, ty + th + 3), max(0, tx - 4):min(w, tx + tw + 4)]
        if cv2.countNonZero(selected_mask) < self.config.min_watermark_area and cv2.countNonZero(text_band) > 40:
            selected_mask[max(0, ty - 3):min(h, ty + th + 3), max(0, tx - 4):min(w, tx + tw + 4)] = text_band

        if cv2.countNonZero(selected_mask) < self.config.min_watermark_area:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        selected_mask = cv2.dilate(selected_mask, kernel, iterations=2)

        self._debug_save("candidate_mask.png", candidate_mask)
        self._debug_save("selected_mask.png", selected_mask)
        return selected_mask

    def _has_watermark(self, roi_bgr: np.ndarray) -> bool:
        return self._build_watermark_mask(roi_bgr) is not None

    # ---------- reconstruction ---------- #

    def _background_fill_from_neighbors(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Fills masked pixels using horizontal/vertical neighbor interpolation.
        This often preserves straight borders and dotted backgrounds better
        than raw inpainting alone.
        """
        out = img_bgr.copy()
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return out

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        expand = self.config.bg_fill_expand
        x0 = max(0, x0 - expand)
        y0 = max(0, y0 - expand)
        x1 = min(w - 1, x1 + expand)
        y1 = min(h - 1, y1 + expand)

        patch = out[y0:y1 + 1, x0:x1 + 1].copy()
        pmask = mask[y0:y1 + 1, x0:x1 + 1]
        ph, pw = pmask.shape[:2]

        # Horizontal interpolation
        horiz = patch.copy().astype(np.float32)
        for yy in range(ph):
            row_mask = pmask[yy] > 0
            if not np.any(row_mask):
                continue
            known = np.where(~row_mask)[0]
            if len(known) < 2:
                continue
            for c in range(3):
                vals = patch[yy, known, c].astype(np.float32)
                interp_idx = np.where(row_mask)[0]
                horiz[yy, interp_idx, c] = np.interp(interp_idx, known, vals)

        # Vertical interpolation
        vert = patch.copy().astype(np.float32)
        for xx in range(pw):
            col_mask = pmask[:, xx] > 0
            if not np.any(col_mask):
                continue
            known = np.where(~col_mask)[0]
            if len(known) < 2:
                continue
            for c in range(3):
                vals = patch[known, xx, c].astype(np.float32)
                interp_idx = np.where(col_mask)[0]
                vert[interp_idx, xx, c] = np.interp(interp_idx, known, vals)

        blend = patch.copy().astype(np.float32)
        masked = pmask > 0
        blend[masked] = 0.5 * horiz[masked] + 0.5 * vert[masked]
        patch_out = np.clip(blend, 0, 255).astype(np.uint8)
        out[y0:y1 + 1, x0:x1 + 1] = patch_out
        return out

    def _patch_reconstruct(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Heals the masked area by copying a nearby clean patch of background.
        This is much better for textures like dotted paper or grain.
        """
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return img_bgr

        # Define the bounding box of the area to heal
        x0, y0, bw, bh = cv2.boundingRect(mask)
        
        # Expand slightly
        pad = 2
        x0_p = max(0, x0 - pad)
        y0_p = max(0, y0 - pad)
        x1_p = min(w, x0 + bw + pad)
        y1_p = min(h, y0 + bh + pad)
        
        bw_p = x1_p - x0_p
        bh_p = y1_p - y0_p

        # Attempt to find a clean source patch
        # Try Left first, then Top
        offsets = [(self.config.patch_offset_x, 0), (0, self.config.patch_offset_y), (self.config.patch_offset_x, self.config.patch_offset_y)]
        
        best_patch = None
        for dx, dy in offsets:
            src_x = x0_p + dx
            src_y = y0_p + dy
            
            # Check if source is within bounds and doesn't overlap too much with mask
            if src_x >= 0 and src_y >= 0 and src_x + bw_p <= w and src_y + bh_p <= h:
                # Check if the source patch itself contains any mask pixels (it should be clean)
                src_mask = mask[src_y:src_y + bh_p, src_x:src_x + bw_p]
                if cv2.countNonZero(src_mask) == 0:
                    best_patch = img_bgr[src_y:src_y + bh_p, src_x:src_x + bw_p].copy()
                    break

        out = img_bgr.copy()
        if best_patch is not None:
            # We have a clean tile. We only paste the pixels where the mask is active
            # to preserve as much original detail as possible.
            target_roi = out[y0_p:y1_p, x0_p:x1_p]
            mask_roi = mask[y0_p:y1_p, x0_p:x1_p]
            
            # Simple alpha blending on the edges of the mask to smooth transition
            mask_float = mask_roi.astype(float) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
            
            for c in range(3):
                target_roi[:, :, c] = (target_roi[:, :, c] * (1 - mask_float) + 
                                       best_patch[:, :, c] * mask_float).astype(np.uint8)
        else:
            # Fallback to inpainting if no clean patch found
            out = cv2.inpaint(out, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

        return out

    def _clean_watermark_in_roi(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Builds a precise mask and removes watermark using patch-based reconstruction.
        """
        mask = self._build_watermark_mask(roi_bgr)
        if mask is None:
            return None

        if self.config.use_patch_heal:
            return self._patch_reconstruct(roi_bgr, mask)
        else:
            return cv2.inpaint(roi_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

    def _inpaint_region(self, img_bgr: np.ndarray) -> np.ndarray:
        # Compatibility method for the PDF strategy 1
        mask = self._build_watermark_mask(img_bgr)
        if mask is None:
            return img_bgr
        return self._patch_reconstruct(img_bgr, mask)

    # ------------------------------------------------------------------ #
    #  PDF processing                                                    #
    # ------------------------------------------------------------------ #

    def _find_watermark_rect_text(self, page: fitz.Page) -> Optional[fitz.Rect]:
        """Locate watermark via PDF text search. Returns padded Rect or None."""
        w, h = page.rect.width, page.rect.height
        instances = page.search_for(self.WATERMARK_TEXT)
        if not instances:
            return None

        best = None
        best_score = float('inf')
        for rect in instances:
            cy = (rect.y0 + rect.y1) / 2
            cx = (rect.x0 + rect.x1) / 2
            if cy < h * 0.78:
                continue
            if cx < w * 0.70:
                continue
            if rect.width > 260 or rect.height > 50:
                continue
            dist = abs(w - cx) + abs(h - cy)
            if dist < best_score:
                best_score = dist
                best = rect

        if best is None:
            return None

        wm_rect = fitz.Rect(best)
        icon_zone = fitz.Rect(best.x0 - 95, best.y0 - 18, best.x0 + 8, best.y1 + 18)
        try:
            for d in page.get_drawings():
                if d.get("rect") and d["rect"].intersects(icon_zone):
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

        wm_rect.x0 = min(wm_rect.x0, best.x0 - 55)
        pad = self.config.watermark_padding
        return fitz.Rect(
            max(0, wm_rect.x0 - pad),
            max(0, wm_rect.y0 - pad),
            min(w, wm_rect.x1 + pad),
            min(h, wm_rect.y1 + pad),
        )

    def _patch_pdf_rect(self, page: fitz.Page, rect: fitz.Rect, precision: bool = True) -> bool:
        mat = fitz.Matrix(self.config.pdf_dpi_scale, self.config.pdf_dpi_scale)
        pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
        roi_bgr = self._pixmap_to_bgr(pix)
        if roi_bgr is None:
            return False

        cleaned = self._clean_watermark_in_roi(roi_bgr) if precision else self._inpaint_region(roi_bgr)
        if cleaned is None:
            return False

        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(cleaned_rgb).save(buf, format='PNG')
        page.insert_image(rect, stream=buf.getvalue(), overlay=True)
        return True

    def process_pdf(self, input_path: str, output_path: str, preview: bool = False) -> bool:
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

            wm_rect = self._find_watermark_rect_text(page)
            patched_now = False
            if wm_rect is not None:
                patched_now = self._patch_pdf_rect(page, wm_rect, precision=True)

            if not patched_now:
                corner = fitz.Rect(
                    max(0, w - self.config.search_margin_x),
                    max(0, h - self.config.search_margin_y),
                    w,
                    h,
                )
                patched_now = self._patch_pdf_rect(page, corner, precision=True)

            if patched_now:
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
    #  Image processing                                                  #
    # ------------------------------------------------------------------ #

    def _clean_roi_scaled(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        scale = self.config.pdf_dpi_scale
        h, w = roi_bgr.shape[:2]
        roi_hr = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cleaned_hr = self._clean_watermark_in_roi(roi_hr)
        if cleaned_hr is None:
            return None
        return cv2.resize(cleaned_hr, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_image(self, input_path: str, output_path: str) -> bool:
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
            img_final = cv2.merge([*cv2.split(img_bgr), alpha]) if has_alpha else img_bgr
            cv2.imwrite(output_path, img_final)
            logger.info(f"Saved cleaned image to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  PPTX processing                                                   #
    # ------------------------------------------------------------------ #

    def _clean_pptx_image_bytes(self, img_bytes: bytes, original_ext: str = ".png") -> Optional[bytes]:
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
        img_final = cv2.merge([*cv2.split(img_bgr), alpha]) if has_alpha else img_bgr

        ext = original_ext.lower()
        if ext in ('.jpg', '.jpeg') and not has_alpha:
            ok, encoded = cv2.imencode('.jpg', img_final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif ext == '.webp':
            ok, encoded = cv2.imencode('.webp', img_final, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
        else:
            ok, encoded = cv2.imencode('.png', img_final)
        return encoded.tobytes() if ok else None

    def process_pptx(self, input_path: str, output_path: str) -> bool:
        tmpdir = None
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
            images = sorted([f for f in os.listdir(media_dir) if f.lower().endswith(image_exts)])
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

                ext = os.path.splitext(img_name)[1]
                cleaned = self._clean_pptx_image_bytes(original, ext)
                if cleaned is not None:
                    with open(img_path, 'wb') as f:
                        f.write(cleaned)
                    patched += 1
                pbar.set_postfix(patched=patched)

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
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
            return False


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Bottom-right watermark remover — PDF, Images & PPTX"
    )
    parser.add_argument("path", help="File (PDF/PPTX/PNG/JPG) or directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--preview", action="store_true", help="Process only first page (PDF only)")
    parser.add_argument("--margin-x", type=int, default=None, help="Search margin width from right edge")
    parser.add_argument("--margin-y", type=int, default=None, help="Search margin height from bottom edge")
    parser.add_argument("--threshold", type=int, default=None, help="Dark contrast threshold")
    parser.add_argument("--text-threshold", type=float, default=None, help="Template match threshold, e.g. 0.48")
    parser.add_argument("--scale", type=float, default=None, help="Render/upscale factor")
    parser.add_argument("--radius", type=int, default=None, help="Inpaint radius")
    parser.add_argument("--no-bg-fill", action="store_true", help="Disable neighbor background reconstruction")
    parser.add_argument("--debug", action="store_true", help="Save debug masks/images")

    args = parser.parse_args()
    config = WatermarkConfig()

    if args.margin_x is not None:
        config.search_margin_x = args.margin_x
    if args.margin_y is not None:
        config.search_margin_y = args.margin_y
    if args.threshold is not None:
        config.pixel_threshold = args.threshold
    if args.text_threshold is not None:
        config.text_match_threshold = args.text_threshold
    if args.scale is not None:
        config.pdf_dpi_scale = args.scale
    if args.radius is not None:
        config.inpaint_radius = args.radius
    if args.no_bg_fill:
        config.use_background_fill = False
    if args.debug:
        config.debug = True

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

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
import base64
from typing import Optional, List, Tuple, Dict
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
    # Search margins from the bottom-right corner.
    # When None, adaptive margins proportional to image dimensions are used (~18% width, ~8% height).
    search_margin_x: Optional[int] = None
    search_margin_y: Optional[int] = None

    # Extra padding around detected watermark bbox (pixels)
    watermark_padding: int = 3

    # Threshold for contrast-based candidate extraction
    pixel_threshold: int = 14

    # PDF rendering scale factor (higher = better quality)
    pdf_dpi_scale: float = 3.5

    # Inpainting radius for cv2.inpaint
    inpaint_radius: int = 3

    # Component filters
    min_watermark_area: int = 20
    min_component_area: int = 4
    max_component_area_ratio: float = 0.20

    # Text and template detection confidence thresholds
    text_match_threshold: float = 0.38
    icon_match_threshold: float = 0.38
    text_luma_threshold: int = 210

    # Reconstruction
    use_patch_heal: bool = True

    # Debug
    debug: bool = False


# ------------------------------------------------------------------ #
#  Core engine                                                       #
# ------------------------------------------------------------------ #

class WatermarkRemover:
    """Removes bottom-right NotebookLM / Gemini watermarks using multi-template detection and healing."""

    WATERMARK_TEXTS = ["Gemini NotebookLM", "Gemini Notebook", "NotebookLM", "Notebook LM"]

    # Canonical 3-arc NotebookLM / Gemini icon template (18x25 png, base64)
    ICON_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAABkAAAASCAAAAACLRSp8AAABkUlEQVQYGXXBMWgTURzA4d+fd8dLuQte6"
        "ImVKFqMm0GFFDJ0qIPUoUMXIVAKFTooOFQaB0EhhYAOChEMKGYQglMcMnTp7lAoxSFIBYdaiKiYYiU57o"
        "7LI9oKOvl9wl8KDP8IR1SKyIByh5HhD+GQEwBn3K/7gBNwRABluLjurr/t5uZmNmeMHhp+E9CxHqzUAWWA"
        "1qXzCgMIOl5YvoLD0F0u+73cT8KJAQZED0u5tfFBrN19eHmtMX+ZjxesyCDaal89EWEVGr5V3GYkiU1iOw"
        "FCP539zKGm96qSZyRfTuraLR3LsWo5nm1HVlTYY2tqa0pV295GP+0E0rwZPOzWgeZmffYg92E7HOt7K2/2"
        "lPTTdPLZAalv76aDcKyTb5V6mfHba1oSOzvfIEZZqaVnPf/AqzzoFK3aDS2JfW7uqY5RRj+68/146Fbv7R"
        "St2qIj4cTZ6bqOUUY/vv8pE/qVuzsFni86ktisPtExyuilF7uTPzILr3cnaV13pFX0C++VAWVGXU53vFOy"
        "Wk6VNpTwP78ABXqQSKynV0oAAAAASUVORK5CYII="
    )

    def __init__(self, config: Optional[WatermarkConfig] = None):
        self.config = config or WatermarkConfig()
        self._template_cache: Dict[Tuple[str, int], np.ndarray] = {}
        icon_bytes = base64.b64decode(self.ICON_B64)
        self._icon_tpl = cv2.imdecode(np.frombuffer(icon_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(self._icon_tpl > 20)
        self._icon_tpl_active = self._icon_tpl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        self._font_path = self._find_system_font()

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

    def get_search_margins(self, width: int, height: int) -> Tuple[int, int]:
        """Compute search margins from right and bottom edges, using adaptive ratios if unspecified."""
        mx = self.config.search_margin_x
        my = self.config.search_margin_y
        if mx is None:
            mx = max(160, int(width * 0.16))
        if my is None:
            my = max(45, int(height * 0.075))
        return min(mx, width), min(my, height)

    def _find_system_font(self) -> Optional[str]:
        candidates = [
            "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
            "/usr/share/fonts/msttcore/arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _render_text_template(self, text: str, height: int) -> np.ndarray:
        """Render binary template for given text string at target height."""
        key = (text, max(6, int(height)))
        if key in self._template_cache:
            return self._template_cache[key]

        font_size = max(10, int(key[1] * 1.15))
        canvas_w = max(120, font_size * (len(text) + 4))
        canvas_h = max(30, font_size * 3)

        img = Image.new('L', (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(img)

        font = None
        if self._font_path:
            try:
                font = ImageFont.truetype(self._font_path, font_size)
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        draw.text((10, 5), text, fill=255, font=font)
        arr = np.array(img)
        ys, xs = np.where(arr > 0)
        if len(xs) == 0 or len(ys) == 0:
            tpl = np.zeros((key[1], key[1] * 5), dtype=np.uint8)
        else:
            tpl = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        self._template_cache[key] = tpl
        return tpl

    def _is_light_background(self, gray: np.ndarray) -> bool:
        """
        Check whether ROI background is light or dark based on edge pixels.
        Edge pixels are sampled from the ROI border ring away from watermark.
        """
        h, w = gray.shape[:2]
        border = max(2, min(h, w) // 20)
        edge_pixels = np.concatenate([
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ])
        return float(np.median(edge_pixels)) >= 128

    # ---------- template matching ---------- #

    def _match_text_and_icon(self, roi_bgr: np.ndarray) -> Optional[dict]:
        """
        Template-matches watermark text and icon glyphs in the ROI with polarity check.
        Returns detection dictionary or None if confidence is below safety thresholds.
        """
        h, w = roi_bgr.shape[:2]
        if h < 12 or w < 40:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        is_light = self._is_light_background(gray)

        best_score = 0.0
        best_box = None
        best_text = None

        min_h = max(5, int(h * 0.04))
        max_h = max(min_h + 3, min(int(h * 0.40), 65))

        # 1. Match full phrases first without histogram equalization
        for text in ["Gemini Notebook", "NotebookLM", "Gemini NotebookLM"]:
            step = 1 if max_h < 30 else 2
            for th in range(min_h, max_h, step):
                tpl = self._render_text_template(text, th)
                if tpl.shape[0] >= h or tpl.shape[1] >= w:
                    continue
                result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score, loc = (-min_val, min_loc) if is_light else (max_val, max_loc)
                if score > best_score:
                    best_score = float(score)
                    best_box = (loc[0], loc[1], tpl.shape[1], tpl.shape[0])
                    best_text = text

        # 2. Fallback only if no full phrase met the safety threshold
        if best_score < self.config.text_match_threshold:
            for th in range(min_h, max_h):
                tpl = self._render_text_template("Notebook", th)
                if tpl.shape[0] >= h or tpl.shape[1] >= w:
                    continue
                result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score, loc = (-min_val, min_loc) if is_light else (max_val, max_loc)
                if score > best_score:
                    best_score = float(score)
                    best_box = (loc[0], loc[1], tpl.shape[1], tpl.shape[0])
                    best_text = "Notebook"

        # safety gate: require confident text match
        if best_box is None or best_score < self.config.text_match_threshold:
            return None

        # corner bias: watermarks sit near the bottom-right corner of the roi
        tx, ty, tw, th = best_box
        if (tx + tw) < int(w * 0.65) or (ty + th) < int(h * 0.55):
            return None

        # 3. match 3-arc icon template to the left of the detected text
        if best_text.startswith("Gemini"):
            ix0 = max(0, tx - int(th * 2.5))
            ix1 = min(w, tx + int(th * 0.2))
        elif best_text == "NotebookLM":
            ix0 = max(0, tx - int(th * 2.5))
            ix1 = min(w, tx + int(th * 0.2))
        else:
            ix0 = max(0, tx - int(th * 3.5))
            ix1 = min(w, tx + int(th * 0.2))

        iy0 = max(0, ty - int(th * 0.4))
        iy1 = min(h, ty + int(th * 1.4))

        best_icon_score = 0.0
        best_icon_box = None
        icon_roi = gray[iy0:iy1, ix0:ix1]
        if icon_roi.shape[0] >= 5 and icon_roi.shape[1] >= 6:
            for ith in range(max(5, th - 4), th + 5):
                for r in (1.25, 1.33, 1.42, 1.53):
                    itw = max(5, int(round(ith * r)))
                    if itw >= icon_roi.shape[1] or ith >= icon_roi.shape[0]:
                        continue
                    itpl = cv2.resize(self._icon_tpl_active, (itw, ith), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(icon_roi, itpl, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    iscore, iloc = (-min_val, min_loc) if is_light else (max_val, max_loc)
                    if iscore > best_icon_score:
                        best_icon_score = float(iscore)
                        best_icon_box = (ix0 + iloc[0], iy0 + iloc[1], itw, ith)

        # fallback single-word notebook requires mandatory icon match
        if best_text == "Notebook":
            if best_icon_box is None or best_icon_score < self.config.icon_match_threshold:
                return None

        # determine glyph bounding box
        gx1 = min(w, tx + tw)
        gy0 = ty
        gy1 = ty + th

        if best_icon_box and best_icon_score >= self.config.icon_match_threshold:
            ix, iy, iw, ih = best_icon_box
            gx0 = ix
            gy0 = min(gy0, iy)
            gy1 = max(gy1, iy + ih)
        else:
            if best_text.startswith("Gemini"):
                gx0 = max(0, tx - int(th * 2.2))
            else:
                gx0 = max(0, tx - int(th * 1.5))

        return {
            "text_box": best_box,
            "text_score": best_score,
            "text": best_text,
            "icon_box": best_icon_box,
            "icon_score": best_icon_score,
            "glyph_box": (gx0, gy0, gx1, gy1),
            "is_light": is_light,
        }

    # ---------- mask building with valley check ---------- #

    def _detect_capsule_bounds(
        self, gray: np.ndarray, gx0: int, gy0: int, gx1: int, gy1: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects whether a translucent pill capsule encloses the glyphs.
        A genuine capsule has distinct boundary step edges on top, bottom, and left.
        Returns (cap_x0, cap_y0, cap_x1, cap_y1) if present, else None.
        """
        h, w = gray.shape[:2]
        gw = gx1 - gx0
        gh = gy1 - gy0
        if gw < 20 or gh < 4:
            return None

        # 1. Top edge: scan upwards looking for first step edge > 10.0
        max_scan_up = max(8, int(gh * 1.5))
        c_col = slice(gx0 + int(gw * 0.1), gx1 - int(gw * 0.1))
        cap_y0 = None
        for y in range(gy0 - 1, max(1, gy0 - max_scan_up), -1):
            d = abs(float(gray[y, c_col].mean()) - float(gray[y - 1, c_col].mean()))
            if d > 10.0:
                cap_y0 = y
                break

        # 2. Bottom edge: scan downwards looking for first step edge > 8.0
        max_scan_bot = max(8, int(gh * 1.5))
        cap_y1 = None
        for y in range(gy1 + 1, min(h - 2, gy1 + max_scan_bot)):
            d = abs(float(gray[y, c_col].mean()) - float(gray[y + 1, c_col].mean()))
            if d > 8.0:
                cap_y1 = y + 1
                break

        # 3. Left edge: scan leftwards looking for first step edge > 10.0
        max_scan_left = max(12, int(gh * 1.8))
        c_row = slice(gy0, gy1)
        cap_x0 = None
        for x in range(gx0 - 1, max(1, gx0 - max_scan_left), -1):
            d = abs(float(gray[c_row, x].mean()) - float(gray[c_row, x - 1].mean()))
            if d > 10.0:
                cap_x0 = x + 1
                break

        # A capsule must have clear boundaries on top, bottom, and left
        if cap_y0 is None or cap_y1 is None or cap_x0 is None:
            return None

        cap_x1 = min(w, gx1 + max(8, int(gh * 0.8)))
        return (cap_x0, cap_y0, cap_x1, cap_y1)

    def _build_watermark_mask(
        self, roi_bgr: np.ndarray, roi_alpha: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Builds a precise mask strictly bounded to the watermark.
        Performs valley checks to prevent touching nearby user text or diagrams.
        Discriminated between naked glyphs and translucent pill capsule.
        """
        h, w = roi_bgr.shape[:2]
        det = self._match_text_and_icon(roi_bgr)
        if det is None:
            return None

        gx0, gy0, gx1, gy1 = det["glyph_box"]
        gw = gx1 - gx0
        gh = gy1 - gy0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        bg_val = float(np.median(gray[max(0, gy0 - 4):min(h, gy1 + 4), max(0, gx0 - 10):min(w, gx1 + 10)]))
        diff = np.abs(gray.astype(float) - bg_val)
        fg_binary = (diff > self.config.pixel_threshold).astype(np.uint8)

        capsule_box = self._detect_capsule_bounds(gray, gx0, gy0, gx1, gy1)
        has_capsule = capsule_box is not None

        mask = np.zeros((h, w), dtype=np.uint8)

        if has_capsule and capsule_box is not None:
            cx0, cy0, cx1, cy1 = capsule_box

            # if alpha channel is present, ensure cx0 covers all transparent watermark pixels
            if roi_alpha is not None:
                band_y0 = max(0, cy0 - 2)
                band_y1 = min(h, cy1 + 2)
                ys_a, xs_a = np.where(roi_alpha[band_y0:band_y1, :] < 255)
                if len(xs_a) > 0:
                    near_wm = (xs_a >= max(0, gx0 - int(gh * 2.5))) & (xs_a <= min(w, gx1 + int(gh * 2.0)))
                    if np.any(near_wm):
                        cx0 = min(cx0, int(xs_a[near_wm].min()))

            # valley check on left to ensure we never cross into user text
            row_start = max(0, cy0 - 2)
            row_end = min(h, cy1 + 2)
            col_counts = np.sum(fg_binary[row_start:row_end, :], axis=0)
            for x in range(gx0, max(0, gx0 - int(gh * 2.5)), -1):
                if col_counts[x] <= 0:
                    cx0 = max(cx0, x + 1)
                    break

            # valley check on top to ensure we never cross into user graphics
            col_start = max(0, cx0 - 5)
            col_end = min(w, cx1 + 5)
            row_counts = np.sum(fg_binary[:, col_start:col_end], axis=1)
            for y in range(gy0, max(0, gy0 - int(gh * 2.0)), -1):
                if row_counts[y] <= 0:
                    cy0 = max(cy0, y + 1)
                    break

            cv2.rectangle(mask, (cx0, cy0), (cx1, cy1), 255, -1)
            if roi_alpha is not None:
                alpha_diff = (roi_alpha < 255).astype(np.uint8) * 255
                nb_mask = np.zeros_like(mask)
                nb_mask[max(0, cy0 - 2):min(h, cy1 + 2), max(0, cx0 - 2):min(w, cx1 + 2)] = 255
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(alpha_diff, nb_mask))
        else:
            # valley check on left for naked glyphs
            row_start = max(0, gy0 - 2)
            row_end = min(h, gy1 + 2)
            col_counts = np.sum(fg_binary[row_start:row_end, :], axis=0)
            x_left_limit = max(0, gx0 - 10)
            for x in range(gx0, max(0, gx0 - max(25, gh * 3)), -1):
                if col_counts[x] <= 0:
                    x_left_limit = x
                    break

            # valley check on top for naked glyphs
            col_start = max(0, gx0 - 15)
            col_end = min(w, gx1 + 15)
            row_counts = np.sum(fg_binary[:, col_start:col_end], axis=1)
            y_top_limit = max(0, gy0 - 4)
            for y in range(gy0, max(0, gy0 - max(20, gh * 3)), -1):
                if row_counts[y] <= max(1, int((col_end - col_start) * 0.02)):
                    y_top_limit = y
                    break

            y_bot_limit = min(h, gy1 + 3)
            box_y0 = max(y_top_limit, gy0 - 1)
            box_y1 = min(y_bot_limit, gy1 + 2)
            box_x0 = max(x_left_limit, gx0 - 2)
            box_x1 = min(w, gx1 + 2)

            box_fg = fg_binary[box_y0:box_y1, box_x0:box_x1].copy()
            if roi_alpha is not None:
                box_fg_alpha = (roi_alpha[box_y0:box_y1, box_x0:box_x1] < 255).astype(np.uint8) * 255
                box_fg = cv2.bitwise_or(box_fg, box_fg_alpha)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            box_fg = cv2.dilate(box_fg, kernel, iterations=1)
            mask[box_y0:box_y1, box_x0:box_x1] = (box_fg > 0).astype(np.uint8) * 255

        self._debug_save("detected_watermark_mask.png", mask)
        return mask, has_capsule

    # ---------- reconstruction ---------- #

    def _patch_reconstruct(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Heals masked area by copying a matching clean patch of nearby background."""
        h, w = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return img_bgr

        x0, y0, bw, bh = cv2.boundingRect(mask)
        pad = 4
        x0_p = max(0, x0 - pad)
        y0_p = max(0, y0 - pad)
        x1_p = min(w, x0 + bw + pad)
        y1_p = min(h, y0 + bh + pad)

        bw_p = x1_p - x0_p
        bh_p = y1_p - y0_p

        dx = max(int(bw_p * 1.2), 24)
        dy = max(int(bh_p * 1.5), 24)
        offsets = [(-dx, 0), (dx, 0), (0, -dy), (0, dy), (-dx, -dy), (dx, -dy)]

        border = min(4, bh_p // 2, bw_p // 2)
        ring = np.zeros((bh_p, bw_p), dtype=bool)
        if border > 0:
            ring[:border, :] = ring[-border:, :] = True
            ring[:, :border] = ring[:, -border:] = True
        dest_ring = img_bgr[y0_p:y1_p, x0_p:x1_p][ring].astype(np.int16)

        best_patch = None
        best_diff = float('inf')
        for ddx, ddy in offsets:
            src_x = x0_p + ddx
            src_y = y0_p + ddy

            if src_x < 0 or src_y < 0 or src_x + bw_p > w or src_y + bh_p > h:
                continue
            src_mask = mask[src_y:src_y + bh_p, src_x:src_x + bw_p]
            if cv2.countNonZero(src_mask) != 0:
                continue

            candidate = img_bgr[src_y:src_y + bh_p, src_x:src_x + bw_p]
            diff = float(np.abs(candidate[ring].astype(np.int16) - dest_ring).mean()) if ring.any() else 0.0
            if diff < best_diff:
                best_diff = diff
                best_patch = candidate.copy()

        out = img_bgr.copy()
        if best_patch is not None and best_diff < 35.0:
            target_roi = out[y0_p:y1_p, x0_p:x1_p]
            mask_roi = mask[y0_p:y1_p, x0_p:x1_p]
            mask_float = cv2.GaussianBlur(mask_roi.astype(float) / 255.0, (3, 3), 0)

            for c in range(3):
                target_roi[:, :, c] = (
                    target_roi[:, :, c] * (1 - mask_float) + best_patch[:, :, c] * mask_float
                ).astype(np.uint8)
        else:
            out = cv2.inpaint(out, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

        return out

    def _clean_watermark_in_roi(
        self, roi_bgr: np.ndarray, roi_alpha: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
        """
        Builds a precise mask and cleans the watermark.
        Returns (cleaned_bgr, cleaned_alpha, mask) or None if no watermark detected.
        """
        res = self._build_watermark_mask(roi_bgr, roi_alpha)
        if res is None:
            return None
        mask, has_capsule = res

        if cv2.countNonZero(mask) < self.config.min_watermark_area:
            return None

        h, w = roi_bgr.shape[:2]

        # Clean BGR
        if has_capsule:
            # 1. Inpaint inner glyphs first to eliminate text contrast inside capsule
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(mask > 0)
            cx0, cx1 = int(xs.min()), int(xs.max() + 1)
            cy0, cy1 = int(ys.min()), int(ys.max() + 1)

            glyph_mask = np.zeros((h, w), dtype=np.uint8)
            glyph_diff = np.abs(gray[cy0:cy1, cx0:cx1].astype(float) - np.median(gray[cy0:cy1, cx0:cx1]))
            glyph_mask[cy0:cy1, cx0:cx1] = (glyph_diff > 16).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            glyph_mask = cv2.dilate(glyph_mask, kernel, iterations=1)

            bgr_intermediate = cv2.inpaint(roi_bgr, glyph_mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

            # 2. sanitize incident user lines on the capsule boundary ring
            if cy0 > 0 and cx1 > cx0:
                bg_top = np.median(bgr_intermediate[cy0 - 1, cx0:cx1], axis=0)
                for x in range(cx0, cx1):
                    if np.linalg.norm(bgr_intermediate[cy0 - 1, x].astype(float) - bg_top) > 45.0:
                        bgr_intermediate[cy0 - 1, x] = bg_top

            if cy1 < h and cx1 > cx0:
                bg_bot = np.median(bgr_intermediate[cy1, cx0:cx1], axis=0)
                for x in range(cx0, cx1):
                    if np.linalg.norm(bgr_intermediate[cy1, x].astype(float) - bg_bot) > 45.0:
                        bgr_intermediate[cy1, x] = bg_bot

            cleaned_bgr = cv2.inpaint(bgr_intermediate, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
            # Strict guarantee: preserve pixels outside mask exactly
            cleaned_bgr[mask == 0] = roi_bgr[mask == 0]
        else:
            if self.config.use_patch_heal:
                cleaned_bgr = self._patch_reconstruct(roi_bgr, mask)
            else:
                cleaned_bgr = cv2.inpaint(roi_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

        # Sanitize alpha channel
        cleaned_alpha = None
        if roi_alpha is not None:
            cleaned_alpha = roi_alpha.copy()
            bg_alpha = 255
            border_alpha = np.concatenate([roi_alpha[0], roi_alpha[-1], roi_alpha[:, 0], roi_alpha[:, -1]])
            if len(border_alpha) > 0:
                bg_alpha = int(np.median(border_alpha))
            cleaned_alpha[mask > 0] = bg_alpha

        return cleaned_bgr, cleaned_alpha, mask

    # ------------------------------------------------------------------ #
    #  PDF processing                                                    #
    # ------------------------------------------------------------------ #

    def _find_watermark_text_rect(self, page: fitz.Page) -> Optional[Tuple[fitz.Rect, str]]:
        """Locates watermark text via PDF text layer. Returns (tight_rect, matched_text)."""
        w, h = page.rect.width, page.rect.height
        best = None
        best_score = float('inf')
        best_text = None

        for text in self.WATERMARK_TEXTS:
            instances = page.search_for(text)
            if not instances:
                continue
            for rect in instances:
                cy = (rect.y0 + rect.y1) / 2
                cx = (rect.x0 + rect.x1) / 2
                if cy < h * 0.70:
                    continue
                if cx < w * 0.50:
                    continue
                if rect.width > 300 or rect.height > 60:
                    continue
                dist = abs(w - cx) + abs(h - cy)
                if dist < best_score:
                    best_score = dist
                    best = rect
                    best_text = text

        if best is None:
            return None

        pad = self.config.watermark_padding
        padded_rect = fitz.Rect(
            max(0, best.x0 - pad),
            max(0, best.y0 - pad),
            min(w, best.x1 + pad),
            min(h, best.y1 + pad),
        )
        return padded_rect, best_text

    def _find_watermark_icon_zone(self, text_rect: fitz.Rect, page_w: float, page_h: float) -> fitz.Rect:
        """Region to the left of the watermark text where its icon glyph sits."""
        span_w = max(45.0, text_rect.height * 2.8)
        span_h = max(18.0, text_rect.height * 1.6)
        return fitz.Rect(
            max(0, text_rect.x0 - span_w),
            max(0, text_rect.y0 - span_h * 0.3),
            min(page_w, text_rect.x0 + 4),
            min(page_h, text_rect.y1 + span_h * 0.3),
        )

    def _redact_pdf_text(self, page: fitz.Page, rect: fitz.Rect, watermark_text: str) -> bool:
        """Deletes watermark text from the PDF text layer via redaction."""
        try:
            page.add_redact_annot(rect, fill=None)
            try:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )
            except TypeError:
                page.apply_redactions()
        except Exception as e:
            logger.debug(f"Redaction failed on page {page.number}: {e}")
            return False
        return not bool(page.search_for(watermark_text))

    def _patch_pdf_rect(self, page: fitz.Page, rect: fitz.Rect) -> bool:
        """Renders rect, detects watermark, and reinserts cleaned sub-region."""
        mat = fitz.Matrix(self.config.pdf_dpi_scale, self.config.pdf_dpi_scale)
        pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
        roi_bgr = self._pixmap_to_bgr(pix)
        if roi_bgr is None:
            return False

        result = self._clean_watermark_in_roi(roi_bgr)
        if result is None:
            return False
        cleaned_bgr, _, mask = result

        x, y, bw, bh = cv2.boundingRect(mask)
        px = max(2, self.config.watermark_padding)
        x0 = max(0, x - px)
        y0 = max(0, y - px)
        x1 = min(roi_bgr.shape[1], x + bw + px)
        y1 = min(roi_bgr.shape[0], y + bh + px)

        scale = self.config.pdf_dpi_scale
        sub_rect = fitz.Rect(
            rect.x0 + x0 / scale, rect.y0 + y0 / scale,
            rect.x0 + x1 / scale, rect.y0 + y1 / scale,
        )

        sub_rgb = cv2.cvtColor(cleaned_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(sub_rgb).save(buf, format='PNG')
        page.insert_image(sub_rect, stream=buf.getvalue(), overlay=True)
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
            patched_now = False

            found = self._find_watermark_text_rect(page)
            if found is not None:
                text_rect, matched_text = found
                icon_zone = self._find_watermark_icon_zone(text_rect, w, h)
                if self._redact_pdf_text(page, text_rect, matched_text):
                    self._patch_pdf_rect(page, icon_zone)
                    patched_now = True
                else:
                    patched_now = self._patch_pdf_rect(page, text_rect | icon_zone)

            if not patched_now:
                mx, my = self.get_search_margins(int(w), int(h))
                corner = fitz.Rect(
                    max(0, w - mx),
                    max(0, h - my),
                    w,
                    h,
                )
                patched_now = self._patch_pdf_rect(page, corner)

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

            mx, my = self.get_search_margins(w, h)
            y0 = max(0, h - my)
            x0 = max(0, w - mx)

            roi_bgr = img_bgr[y0:h, x0:w].copy()
            roi_alpha = alpha[y0:h, x0:w].copy() if has_alpha else None

            result = self._clean_watermark_in_roi(roi_bgr, roi_alpha)
            if result is None:
                logger.warning(f"No watermark detected in {input_path}")
                return False

            cleaned_bgr, _, _ = result
            img_bgr[y0:h, x0:w] = cleaned_bgr
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

        mx, my = self.get_search_margins(w, h)
        y0, x0 = max(0, h - my), max(0, w - mx)

        roi_bgr = img_bgr[y0:h, x0:w].copy()
        roi_alpha = alpha[y0:h, x0:w].copy() if has_alpha else None

        result = self._clean_watermark_in_roi(roi_bgr, roi_alpha)
        if result is None:
            return None

        cleaned_bgr, _, _ = result
        img_bgr[y0:h, x0:w] = cleaned_bgr
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
#  CLI                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Bottom-right watermark remover - PDF, Images & PPTX"
    )
    parser.add_argument("path", help="File (PDF/PPTX/PNG/JPG) or directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--preview", action="store_true", help="Process only first page (PDF only)")
    parser.add_argument("--margin-x", type=int, default=None, help="Search margin width from right edge (adaptive if omitted)")
    parser.add_argument("--margin-y", type=int, default=None, help="Search margin height from bottom edge (adaptive if omitted)")
    parser.add_argument("--threshold", type=int, default=None, help="Contrast threshold (default: 14)")
    parser.add_argument("--text-threshold", type=float, default=None, help="Template match threshold (default: 0.38)")
    parser.add_argument("--scale", type=float, default=None, help="Render/upscale factor (default: 3.5)")
    parser.add_argument("--radius", type=int, default=None, help="Inpaint radius (default: 3)")
    parser.add_argument("--no-patch-heal", action="store_true", help="Disable clean-patch healing, use plain inpainting instead")
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
    if args.no_patch_heal:
        config.use_patch_heal = False
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

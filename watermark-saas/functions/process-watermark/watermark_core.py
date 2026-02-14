#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core watermark removal logic extracted from remover.py.
Used by the Lambda process-watermark function.
"""

import cv2
import numpy as np
import logging
import os
import io
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# PDF processing strategy constants
PDF_STRATEGY_AUTO = "auto"
PDF_STRATEGY_REPLACE = "replace"
PDF_STRATEGY_OVERLAY = "overlay"


@dataclass
class WatermarkConfig:
    """Configuration for watermark detection and removal."""
    # Search area
    search_margin_x: int = 200
    search_margin_y: int = 50

    # Vector detection limits
    vector_max_width: int = 150

    # PDF Image Quality (only used in overlay mode)
    pdf_dpi_scale: float = 2.0

    # Inpainting settings
    inpaint_radius: int = 3

    # Detection settings
    blur_kernel_size: int = 7
    binary_threshold: int = 50
    max_component_ratio: float = 0.8
    left_protect_ratio: float = 0.2

    # PDF processing strategy: "auto", "replace", or "overlay"
    pdf_strategy: str = PDF_STRATEGY_AUTO


class WatermarkRemover:
    def __init__(self, config: WatermarkConfig = WatermarkConfig()):
        self.config = config

    def _clean_image_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Core logic: Takes a BGR numpy array (image chunk), detects the watermark
        using local contrast (blur difference), and inpaints it.
        """
        try:
            blurred = cv2.medianBlur(img_bgr, self.config.blur_kernel_size)

            diff = cv2.absdiff(img_bgr, blurred)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(diff_gray, self.config.binary_threshold, 255, cv2.THRESH_BINARY)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            h, w = mask.shape

            new_mask = np.zeros_like(mask)

            for i in range(1, num_labels):
                x, y, stat_w, stat_h, area = stats[i]

                if stat_w > w * self.config.max_component_ratio or stat_h > h * self.config.max_component_ratio:
                    continue
                if x < w * self.config.left_protect_ratio:
                    continue
                if x == 0 or y == 0:
                    continue

                new_mask[labels == i] = 255

            mask = new_mask

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)

            if cv2.countNonZero(mask) == 0:
                return img_bgr

            cleaned = cv2.inpaint(img_bgr, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)

            return cleaned
        except Exception as e:
            logger.warning(f"Inpainting failed: {e}")
            return img_bgr

    def process_image(self, input_path: str, output_path: str) -> bool:
        """Processes an image using Gradient-Aware Inpainting, preserving Alpha."""
        try:
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Could not read image: {input_path}")
                return False

            h, w = img.shape[:2]

            has_alpha = False
            if len(img.shape) == 3 and img.shape[2] == 4:
                has_alpha = True
                b, g, r, a = cv2.split(img)
                img_bgr = cv2.merge([b, g, r])
            else:
                img_bgr = img

            margin_x = self.config.search_margin_x
            margin_y = self.config.search_margin_y
            x_start = max(0, w - margin_x)
            y_start = max(0, h - margin_y)

            roi_bgr = img_bgr[y_start:h, x_start:w]

            cleaned_roi = self._clean_image_array(roi_bgr)

            img_bgr[y_start:h, x_start:w] = cleaned_roi

            if has_alpha:
                img_final = cv2.merge([*cv2.split(img_bgr), a])
            else:
                img_final = img_bgr

            cv2.imwrite(output_path, img_final)
            logger.info(f"Saved cleaned image to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing image {input_path}: {e}")
            return False

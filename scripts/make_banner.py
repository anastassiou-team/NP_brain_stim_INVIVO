#!/usr/bin/env python3
"""
Generate a README banner image by compositing key figure panels into a
single wide row.

Panels (left to right):
  1. Fig 1f  – bipolar Ve analytical model
  2. Fig 3a-b – VL clustering scatter + bubble chart
  3. Fig 4b  – brain-wide firing-rate changes
  4. Fig 4e  – FS vs RS z-scored rate by distance

Usage:
    python scripts/make_banner.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from config.paths import FIGURES_OUTPUT

PANELS = [
    'figure1f_bipolar_model.png',
    'figure3_row1.png',
    'figure4b_brain_wide_fr.png',
    'figure4e_zscore_distance.png',
]

BANNER_HEIGHT = 600          # pixels
PADDING = 20                 # gap between panels (pixels)
BG_COLOR = (255, 255, 255)   # white


def main():
    images = []
    for name in PANELS:
        path = os.path.join(FIGURES_OUTPUT, name)
        if not os.path.exists(path):
            print(f'Missing {path} – run the figure scripts first.')
            sys.exit(1)
        images.append(Image.open(path).convert('RGB'))

    # scale each to the target height, preserving aspect ratio
    resized = []
    for img in images:
        scale = BANNER_HEIGHT / img.height
        new_w = int(img.width * scale)
        resized.append(img.resize((new_w, BANNER_HEIGHT), Image.LANCZOS))

    total_width = sum(im.width for im in resized) + PADDING * (len(resized) - 1)
    banner = Image.new('RGB', (total_width, BANNER_HEIGHT), BG_COLOR)

    x = 0
    for im in resized:
        banner.paste(im, (x, 0))
        x += im.width + PADDING

    out = os.path.join(FIGURES_OUTPUT, 'banner.png')
    banner.save(out, dpi=(150, 150))
    print(f'Saved {out}  ({total_width}×{BANNER_HEIGHT} px)')


if __name__ == '__main__':
    main()

"""
Generates sample_image.png — a synthetic "scanned document" with text,
slight noise, and uneven lighting, so the pre-processing pipeline
(grayscale -> blur -> adaptive threshold) has something real to clean up.

Run once: python3 generate_sample.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 800, 400


def build_sample():
    # Base canvas with a subtle lighting gradient (simulates uneven scan lighting)
    gradient = np.tile(np.linspace(235, 255, WIDTH), (HEIGHT, 1)).astype(np.uint8)
    img = Image.fromarray(gradient).convert("RGB")

    draw = ImageDraw.Draw(img)

    font = None
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(candidate, 42)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    lines = [
        "DecodeLabs AI Project 4",
        "Optical Character Recognition",
        "Confidence Threshold: 80%",
    ]
    y = 60
    for line in lines:
        draw.text((60, y), line, fill=(20, 20, 20), font=font)
        y += 90

    # Add a bit of speckle noise to simulate scan artifacts
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(arr)

    noisy_img.save("sample_image.png")
    print("Created sample_image.png")


if __name__ == "__main__":
    build_sample()

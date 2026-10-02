"""
Text Recognition Engine
------------------------
DecodeLabs AI Internship — Project 4: Image / Text Recognition (Basic)

A basic OCR (Optical Character Recognition) pipeline that ingests a raw
image, cleans it up with classic computer-vision pre-processing, extracts
machine-readable text with pytesseract (Google's Tesseract OCR engine),
and filters results through an 80% confidence gate before presenting
the output — both as formatted text and as an annotated image with
bounding boxes drawn around every accepted word.

Pipeline (IPO model):
    1. Ingestion      : load the raw image
    2. Pre-processing : grayscale -> Gaussian blur -> adaptive threshold
    3. Recognition     : run Tesseract OCR, word-by-word, with confidence
    4. Filtering       : keep only words with confidence >= CONFIDENCE_THRESHOLD
    5. Output          : print the clean text + save an annotated image
                          with bounding boxes and confidence labels
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

CONFIDENCE_THRESHOLD = 80  # The Gatekeeper Rule: 80% minimum validated confidence
DEFAULT_IMAGE = Path(__file__).parent / "sample_image.png"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_image(path: Path) -> np.ndarray:
    """Ingestion: load the raw image from disk."""
    if not path.exists():
        sys.exit(f"Error: image not found at {path}")
    image = cv2.imread(str(path))
    if image is None:
        sys.exit(f"Error: could not read image at {path} (unsupported format?)")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Pre-processing: collapse to grayscale, smooth noise with a Gaussian
    blur, then force a clean binary image with adaptive thresholding so
    Tesseract sees crisp, high-contrast character shapes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return thresh


def recognize(preprocessed: np.ndarray) -> list[dict]:
    """
    Recognition + Filtering: run Tesseract word-by-word with confidence
    scores, keeping only words at or above the confidence threshold
    (the "Gatekeeper Rule").
    """
    raw = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)

    results = []
    for i, text in enumerate(raw["text"]):
        word = text.strip()
        confidence = int(raw["conf"][i]) if raw["conf"][i] != "-1" else -1

        if not word or confidence < CONFIDENCE_THRESHOLD:
            continue

        results.append(
            {
                "text": word,
                "confidence": confidence,
                "x": raw["left"][i],
                "y": raw["top"][i],
                "w": raw["width"][i],
                "h": raw["height"][i],
            }
        )
    return results


def annotate(original: np.ndarray, results: list[dict]) -> np.ndarray:
    """Output (visual): draw a bounding box + confidence label for every accepted word."""
    annotated = original.copy()
    for word in results:
        x, y, w, h = word["x"], word["y"], word["w"], word["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f'{word["text"]} ({word["confidence"]}%)'
        cv2.putText(
            annotated,
            label,
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 120, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def display_results(results: list[dict]):
    print(f"\nText Recognition Results (confidence >= {CONFIDENCE_THRESHOLD}%)")
    print("-" * 55)

    if not results:
        print("No text passed the confidence gate.")
        return

    full_text = " ".join(word["text"] for word in results)
    avg_conf = round(sum(w["confidence"] for w in results) / len(results), 1)

    print(f"Recognized text:\n  {full_text}\n")
    print(f"Words accepted : {len(results)}")
    print(f"Average confidence: {avg_conf}%")

    print("\nPer-word breakdown:")
    for word in results:
        print(f'  "{word["text"]}" — {word["confidence"]}%')


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    print(f"Loading image: {image_path}")

    original = load_image(image_path)
    preprocessed = preprocess(original)
    results = recognize(preprocessed)
    display_results(results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    annotated = annotate(original, results)
    out_path = OUTPUT_DIR / "annotated_output.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated image saved to: {out_path}")


if __name__ == "__main__":
    main()

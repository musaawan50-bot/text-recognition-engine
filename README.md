# Text Recognition Engine (OCR)

A basic Optical Character Recognition (OCR) pipeline — built for
**DecodeLabs AI Internship, Project 4: Image / Text Recognition**.

It ingests a raw image, cleans it up with classic computer-vision
pre-processing, extracts machine-readable text using **pytesseract**
(Google's Tesseract OCR engine), and filters results through an
**80% confidence gate** before presenting the output as both clean text
and an annotated image with bounding boxes.

## How it works (IPO Pipeline)

1. **Ingestion** — load the raw image from disk
2. **Pre-processing** — grayscale conversion → Gaussian blur → adaptive
   thresholding, to strip out shadows, noise, and uneven lighting before
   recognition
3. **Recognition** — run Tesseract OCR word-by-word, each with its own
   confidence score
4. **Filtering (the Gatekeeper Rule)** — discard any word below an 80%
   confidence score, so low-certainty guesses never reach the output
5. **Output** — print the recognized text + per-word confidence, and
   save an annotated image with a green bounding box and confidence
   label drawn over every accepted word

## Project structure

```
text-recognition-engine/
├── recognizer.py         # main script — the OCR pipeline
├── generate_sample.py    # creates sample_image.png (run once)
├── sample_image.png      # sample input with text + simulated noise
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (VS Code)

**1. Install the Tesseract OCR engine itself** (pytesseract is just a
Python wrapper — it needs the actual Tesseract binary installed on your
system):

- **Mac:** `brew install tesseract`
- **Windows:** download and run the installer from the
  [Tesseract at UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  page, then make sure its install folder is added to your PATH
- **Linux:** `sudo apt install tesseract-ocr`

**2. Open the project folder in VS Code** (File → Open Folder…).

**3. In the VS Code terminal**, set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

**4. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**5. Run it on the included sample image:**
```bash
python3 recognizer.py
```

You'll see the recognized text printed with per-word confidence scores,
and an annotated image saved to `output/annotated_output.png`.

**6. Try it on your own image:**
```bash
python3 recognizer.py path/to/your/image.png
```

## Extending it

- Lower or raise `CONFIDENCE_THRESHOLD` in `recognizer.py` to see how it
  trades off false positives vs. false negatives.
- Swap in `--psm` tuning (see `pytesseract.image_to_data(..., config="--psm 6")`)
  for different layouts — single text blocks, number plates, invoices.
- Add the Path 2 extension: object detection with OpenCV's `cv2.dnn` and
  a pre-trained MobileNet-SSD model, applying the same 80% confidence
  gate to bounding boxes instead of words.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Text Recognition Engine (OCR pipeline)"
git branch -M main
git remote add origin https://github.com/<your-username>/text-recognition-engine.git
git push -u origin main
```

(Create the empty repo on GitHub first — github.com → New repository —
then paste its URL into the `git remote add` command above.)

## Posting on LinkedIn

A short outline you can adapt:

> 🚀 Just completed Project 4 of my AI Engineering track at DecodeLabs —
> a **Text Recognition Engine** that turns raw images into
> machine-readable text using classic computer vision + OCR.
>
> Key concepts I implemented:
> ✅ A full image pre-processing pipeline: grayscale → Gaussian blur →
> adaptive thresholding, to clean up noise and uneven lighting
> ✅ Optical Character Recognition with Tesseract via pytesseract
> ✅ An 80% confidence gate — filtering out low-certainty guesses before
> they ever reach the output
> ✅ Visual confirmation: bounding boxes + confidence scores drawn
> directly on the image
>
> Code on GitHub: [link]
> #AI #ComputerVision #OCR #Python #DecodeLabs

Attach a screenshot of the terminal output next to the annotated image
(`output/annotated_output.png`) — the visual pairing of "raw numbers" +
"clean picture" tends to land well.

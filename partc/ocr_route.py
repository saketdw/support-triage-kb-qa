"""Route customer screenshots into the same four support routes (Part C).

    python -m partc.ocr_route --input media/screenshots --output routed.csv

Architecture: **one router, many input adapters.** The screenshot is OCR'd
locally (Tesseract) and the text goes through the *same* classifier as chat
and email — no second model, no separate training data; every improvement to
the router benefits every channel.

Why local OCR and not a hosted vision API: these images contain an email
address, a **live one-time code**, a wallet address and an account balance.
Sending them to a third-party API is a data-processing event and deposits an
active credential in someone else's request logs. Local processing costs ~$0
and ~200 ms per ticket. Raw OCR text is written only to the output the
operator asks for — in production it would be redacted (OTP-shaped digit
runs, wallet addresses) before touching any log.

Degradation gates, because real-world screenshots are photos of screens,
cropped and recompressed: mean per-word OCR confidence and a minimum word
count; below either gate the ticket goes to a human with the image attached.
The subtler failure is covariate shift — OCR output (UI chrome + prose) is
not the distribution the classifier was trained on — so the route confidence
is reported too, and the review band catches what OCR confidence cannot.
Measured on the three provided images: all route correctly; the phishing SMS
routes at 0.31 confidence — correct but visibly unsure, exactly the ticket a
review band should flag. Dark-mode inversion is applied when luminance says
so; on these clean renders it measured neutral (Tesseract 5 copes), kept as
no-cost insurance for low-contrast real captures.

This is an optional extra: it needs `brew install tesseract` (or apt
tesseract-ocr) and `pip install -r requirements-partc.txt`.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

from triage.model import build_pipeline, load_training

OCR_CONF_MIN = 60.0   # mean per-word Tesseract confidence below this -> human
MIN_WORDS = 5         # fewer OCR'd words than this -> something is wrong
ROUTE_CONF_REVIEW = 0.70  # classifier confidence below this -> flag for review


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def ocr_image(path: Path) -> tuple[str, float, int]:
    """OCR one image -> (text, mean word confidence, word count)."""
    from PIL import Image, ImageOps
    import pytesseract

    img = Image.open(path).convert("L")
    if np.array(img).mean() < 128:  # dark theme: invert to dark-on-light
        img = ImageOps.invert(img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    pairs = [
        (w.strip(), int(c))
        for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) >= 0
    ]
    if not pairs:
        return "", 0.0, 0
    words, confs = zip(*pairs)
    return " ".join(words), float(np.mean(confs)), len(words)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a directory of images, or one image file")
    ap.add_argument("--output", required=True, help="where to write the routing CSV")
    args = ap.parse_args(argv)

    if shutil.which("tesseract") is None:
        _fail("tesseract binary not found - Part C is an optional extra: "
              "`brew install tesseract` (macOS) or `apt-get install tesseract-ocr`, "
              "then `pip install -r requirements-partc.txt`")
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        _fail("missing Python deps - run `pip install -r requirements-partc.txt`")

    src = Path(args.input)
    if not src.exists():
        _fail(f"input not found: {src}")
    images = sorted(p for p in ([src] if src.is_file() else src.iterdir())
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        _fail(f"no images found in {src}")

    texts, labels = load_training()
    pipe = build_pipeline().fit(texts, labels)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "ocr_confidence", "words", "route", "route_confidence", "needs_review"])
        for img_path in images:
            text, ocr_conf, n_words = ocr_image(img_path)
            if ocr_conf < OCR_CONF_MIN or n_words < MIN_WORDS:
                writer.writerow([img_path.name, f"{ocr_conf:.1f}", n_words, "human-triage", "", "true"])
                continue
            route = pipe.predict([text])[0]
            route_conf = float(pipe.predict_proba([text]).max())
            review = route_conf < ROUTE_CONF_REVIEW
            writer.writerow([img_path.name, f"{ocr_conf:.1f}", n_words, route,
                             f"{route_conf:.2f}", str(review).lower()])
    print(f"routed {len(images)} image(s) to {args.output}")


if __name__ == "__main__":
    main()

import re
import fitz
import pandas as pd

from pathlib import Path


# =========================================================
# Settings
# =========================================================

PDF_DIR = Path("ijo_pdfs")
OUTPUT_DIR = Path("ijo_pre_discussion")
LOG_FILE = "ijo_text_preparation_log.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Extract PDF text
# =========================================================

def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text("text")

        pages.append(text)

    doc.close()

    return "\n".join(pages)


# =========================================================
# Find Discussion heading
# =========================================================

def find_discussion(text):

    patterns = [

        # 4 Discussion
        # 4. Discussion
        # 5 DISCUSSION
        r"(?im)^\s*\d+(?:\.\d+)*\.?\s+discussion\s*$",

        # Discussion
        r"(?im)^\s*discussion\s*$",

        # Discussion and Conclusions
        r"(?im)^\s*\d+(?:\.\d+)*\.?\s+discussion\s+and\s+conclusions?\s*$",

        r"(?im)^\s*discussion\s+and\s+conclusions?\s*$",

        # Discussion and Conclusion
        r"(?im)^\s*\d+(?:\.\d+)*\.?\s+discussion\s+and\s+conclusion\s*$",

    ]

    matches = []

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            matches.append(match)

    if not matches:
        return None

    # use earliest matching Discussion heading
    return min(
        matches,
        key=lambda x: x.start()
    )


# =========================================================
# Clean text
# =========================================================

def clean_text(text):

    # normalize common PDF spacing artifacts
    text = text.replace("\u2009", " ")
    text = text.replace("\u00a0", " ")

    # remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# Process PDFs
# =========================================================

pdf_files = list(
    PDF_DIR.glob("*.pdf")
)

print(
    f"PDFs found: {len(pdf_files)}"
)

results = []


for i, pdf_path in enumerate(pdf_files, start=1):

    print()
    print(
        f"[{i}/{len(pdf_files)}] "
        f"{pdf_path.name}"
    )

    try:

        full_text = extract_pdf_text(
            pdf_path
        )

        full_text = clean_text(
            full_text
        )

        discussion_match = find_discussion(
            full_text
        )

        if discussion_match:

            output_text = full_text[
                :discussion_match.start()
            ].strip()

            status = "discussion_found"

            discussion_heading = (
                discussion_match.group(0)
                .strip()
            )

            print(
                "Discussion found:",
                discussion_heading
            )

        else:

            # Do NOT silently use the full paper.
            # Save separately for manual inspection.
            output_text = full_text

            status = "discussion_not_found"

            discussion_heading = ""

            print(
                "WARNING: Discussion heading not found"
            )


        output_path = (
            OUTPUT_DIR /
            f"{pdf_path.stem}.txt"
        )

        output_path.write_text(
            output_text,
            encoding="utf-8"
        )

        results.append({
            "pdf_file": pdf_path.name,
            "txt_file": output_path.name,
            "status": status,
            "discussion_heading": discussion_heading,
            "full_text_characters": len(full_text),
            "output_characters": len(output_text)
        })

        print(
            "Characters:",
            len(full_text),
            "→",
            len(output_text)
        )

    except Exception as e:

        print(
            "ERROR:",
            type(e).__name__,
            str(e)
        )

        results.append({
            "pdf_file": pdf_path.name,
            "txt_file": "",
            "status": "error",
            "discussion_heading": "",
            "full_text_characters": None,
            "output_characters": None
        })


# =========================================================
# Save log
# =========================================================

log_df = pd.DataFrame(
    results
)

log_df.to_csv(
    LOG_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\nFinished.")

print(
    log_df["status"]
    .value_counts(dropna=False)
)

print(
    f"\nSaved TXT files to: "
    f"{OUTPUT_DIR}"
)

print(
    f"Saved log: {LOG_FILE}"
)

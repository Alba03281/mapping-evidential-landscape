import os
import time
import pandas as pd

from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


# =========================================================
# 1. Settings
# =========================================================

INPUT_FILE = "ijo_china_candidates.csv"
OUTPUT_FILE = "ijo_human_remains_screened.csv"

MODEL = "gemini-3.1-flash-lite"

SLEEP_SECONDS = 0.5


# =========================================================
# 2. Output schema
# =========================================================

class ScreeningResult(BaseModel):

    relevance: Literal[
        "relevant",
        "not_relevant",
        "uncertain"
    ]

    relevance_reason: str


# =========================================================
# 3. Load Gemini API
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )

client = genai.Client(
    api_key=api_key
)


# =========================================================
# 4. Load candidate articles
# =========================================================

df = pd.read_csv(INPUT_FILE)

print(f"Candidates loaded: {len(df)}")


# =========================================================
# 5. Load previous progress if available
# =========================================================

if os.path.exists(OUTPUT_FILE):

    previous = pd.read_csv(OUTPUT_FILE)

    completed_dois = set(
        previous["doi"]
        .dropna()
        .astype(str)
    )

    print(
        f"Already screened: {len(completed_dois)}"
    )

    results = previous.to_dict(
        orient="records"
    )

else:

    completed_dois = set()
    results = []


# =========================================================
# 6. Screening function
# =========================================================

def screen_article(title, abstract):

    title = "" if pd.isna(title) else str(title)
    abstract = "" if pd.isna(abstract) else str(abstract)

    prompt = f"""
You are screening publications for a database of archaeological
and historical human remains from China.

Your task is ONLY to determine whether this publication is relevant
to the database based on the TITLE and ABSTRACT supplied below.

CLASSIFICATION:

"relevant":
Return relevant if the publication directly studies archaeological
or historical HUMAN REMAINS from a site or cemetery located in China.

Relevant human remains may include:
- human skeletons
- human bones
- human teeth
- skeletal populations
- mummified human remains
- paleopathology
- bioarchaeology
- human osteology
- trauma in archaeological human remains
- dental pathology
- entheseal changes
- ancient DNA from archaeological human remains
- stable isotope analysis of archaeological human remains
- proteomic or other biomolecular analysis of human remains
- case studies of archaeological human individuals

The human remains must be part of the actual study.

"not_relevant":
Return not_relevant if:
- China is mentioned only as background or comparison
- the archaeological human remains are from outside China
- the study concerns animals only
- the study concerns archaeological artifacts but does not directly
  analyse human remains
- the study concerns plants or environmental remains only
- it is modern clinical research
- it is modern forensic research without archaeological or historical
  human remains
- human remains are mentioned only in cited literature
- the study uses only previously published Chinese human-remains data
  without directly analysing archaeological or historical human remains
  as part of the present research

"uncertain":
Return uncertain when the title and abstract do not provide enough
information to determine whether archaeological or historical human
remains from China are directly studied.

IMPORTANT:
- Do not infer information not present in the title or abstract.
- If the abstract is missing or too vague, prefer "uncertain".
- A publication does NOT need to use the words "bioarchaeology" or
  "human remains" if the abstract clearly states that ancient human
  bones, skeletons, teeth, DNA, isotopes, or other human skeletal
  material from China were directly analysed.

relevance_reason:
Give one short sentence explaining the classification.
Do not provide a long explanation.


TITLE:
{title}


ABSTRACT:
{abstract}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScreeningResult,
            temperature=0
        )
    )

    if response.parsed is not None:

        result = response.parsed

        if isinstance(result, ScreeningResult):
            return result.model_dump()

        return result

    raise ValueError(
        "Gemini returned no parsed result."
    )


# =========================================================
# 7. Screen candidates
# =========================================================

for index, row in df.iterrows():

    doi = str(row.get("doi", ""))

    # Skip already completed DOI
    if doi in completed_dois:
        continue

    title = row.get("title", "")
    abstract = row.get("abstract", "")

    print()
    print(
        f"[{index + 1}/{len(df)}] "
        f"{title[:100]}"
    )

    try:

        result = screen_article(
            title,
            abstract
        )

        output_row = row.to_dict()

        output_row["relevance"] = (
            result["relevance"]
        )

        output_row["relevance_reason"] = (
            result["relevance_reason"]
        )

        results.append(output_row)

        print(
            "→",
            result["relevance"],
            "|",
            result["relevance_reason"]
        )

    except Exception as e:

        print(
            "ERROR:",
            type(e).__name__,
            str(e)
        )

        output_row = row.to_dict()

        output_row["relevance"] = "uncertain"

        output_row["relevance_reason"] = (
            f"Screening error: {type(e).__name__}"
        )

        results.append(output_row)


    # =====================================================
    # Save after EVERY article
    # =====================================================

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    time.sleep(SLEEP_SECONDS)


# =========================================================
# 8. Final summary
# =========================================================

output_df = pd.DataFrame(results)

print("\nFinished.")

print(
    output_df["relevance"]
    .value_counts(dropna=False)
)

print(
    f"\nSaved: {OUTPUT_FILE}"
)
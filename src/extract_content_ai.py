"""
AI-assisted content extraction for bioarchaeology / paleopathology publications.

Public repository notes
-----------------------
- API credentials are loaded from environment variables via `.env`.
  Never commit `.env` or API keys to a public repository.
- Input PDFs/TXT files and generated outputs may contain copyrighted
  publication text or extracted evidence. Keep them out of version control
  unless you have permission to share them.
- This is a pilot / experimental content-level extractor. Outputs should be
  manually reviewed before being used as research data.
"""

import os
import json
import time
import pandas as pd

from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


# =========================================================
# Settings
# =========================================================

INPUT_SOURCES = {
    "IJO": Path("ijo_pre_discussion"),
    "IJPP": Path("ijpp_pre_discussion"),
    "AJPA_AJBA_AJHB": Path("ajpa_ajba_ajhb_pre_discussion")
}

SCREENING_FILE = "ajpa_ajba_ajhb_human_remains_screened.csv"

RAW_JSON_DIR = Path("raw_json")
RAW_JSON_DIR.mkdir(exist_ok=True)

PUBLICATIONS_FILE = "publications.csv"
SITES_FILE = "sites.csv"
ERROR_FILE = "extraction_errors.csv"

MAX_RETRIES = 3

MANUAL_JOURNAL_MAP = {
    "AJPA_old_001": "AJPA",
    "AJPA_old_002": "AJPA",
    "AJPA_old_003": "AJPA",
}


# =========================================================
# Helper: convert TXT filename back to DOI
# =========================================================

def paper_id_to_doi(paper_id):

    if paper_id.startswith("10."):
        return paper_id.replace("_", "/", 1).lower()

    return ""


# =========================================================
# Build DOI -> journal lookup
# =========================================================

screening_df = pd.read_csv(
    SCREENING_FILE
)

screening_df["doi_clean"] = (
    screening_df["doi"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(
        "https://doi.org/",
        "",
        regex=False
    )
    .str.replace(
        "http://doi.org/",
        "",
        regex=False
    )
    .str.replace(
        "doi:",
        "",
        regex=False
    )
    .str.strip()
)

doi_to_journal = dict(
    zip(
        screening_df["doi_clean"],
        screening_df["journal"]
    )
)

print(
    f"DOI-journal lookup records: "
    f"{len(doi_to_journal)}"
)

# =========================================================
# AJPA / AJBA / AJHB journal lookup
# =========================================================

screening_df = pd.read_csv(SCREENING_FILE)

screening_df["doi_clean"] = (
    screening_df["doi"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("https://doi.org/", "", regex=False)
    .str.replace("http://doi.org/", "", regex=False)
)

doi_to_journal = dict(
    zip(
        screening_df["doi_clean"],
        screening_df["journal"]
    )
)



# =========================================================
# Collect TXT files from all journals
# =========================================================

txt_files = []

for source_name, input_dir in INPUT_SOURCES.items():

    if not input_dir.exists():

        print(
            "Folder not found:",
            input_dir
        )

        continue

    for txt_path in sorted(
    input_dir.glob("*.txt")
):

        if source_name == "AJPA_AJBA_AJHB":

            paper_id = txt_path.stem

            if paper_id in MANUAL_JOURNAL_MAP:

                journal = MANUAL_JOURNAL_MAP[
                paper_id
                ]

            else:

                doi = paper_id_to_doi(
                paper_id
                )

                journal = doi_to_journal.get(
                doi,
                "UNKNOWN"
                )

        else:

            journal = source_name

        txt_files.append(
            (
            journal,
            txt_path
            )
        )

print(
    f"Total TXT files: {len(txt_files)}"
)

unknown_files = [
    txt_path.name
    for journal, txt_path in txt_files
    if journal == "UNKNOWN"
]

print(
    f"UNKNOWN journal files: "
    f"{len(unknown_files)}"
)

if unknown_files:

    print(
        "\nFiles with UNKNOWN journal:"
    )

    for filename in unknown_files:
        print(filename)
        

def get_completed_ids():

    if not os.path.exists(PUBLICATIONS_FILE):
        return set()

    df = pd.read_csv(
        PUBLICATIONS_FILE
    )

    if "publication_id" not in df.columns:
        return set()

    # Old database may not yet have journal column
    if "journal" not in df.columns:
        df["journal"] = "IJO"

    else:
        df["journal"] = (
            df["journal"]
            .fillna("IJO")
            .replace("", "IJO")
        )

    return set(
        zip(
            df["journal"]
            .astype(str),

            df["publication_id"]
            .fillna("")
            .astype(str)
        )
    )

def call_gemini_with_retry(prompt):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                    temperature=0
                )
            )

            return response

        except Exception as e:

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed:",
                type(e).__name__,
                str(e)
            )

            if attempt < MAX_RETRIES:
                time.sleep(10)

            else:
                raise



# =========================================================
# 1. Define output schema
# =========================================================

class SiteInfo(BaseModel):
    
    site_name: str
    site_name_evidence: Optional[str] = None

    site_location: Optional[str] = None
    site_location_evidence: Optional[str] = None

    absolute_date: Optional[str] = None
    absolute_date_evidence: Optional[str] = None

    period: Optional[str] = None
    period_evidence: Optional[str] = None

    culture: Optional[str] = None
    culture_evidence: Optional[str] = None

    analysis_successful: Optional[bool] = None
    analysis_status_evidence: Optional[str] = None

class ExtractionResult(BaseModel):

    title: Optional[str] = None
    publication_year: Optional[int] = None

    authors: Optional[List[str]] = None
    author_affiliations: Optional[List[str]] = None

    study_scope: Optional[str] = None
    study_scope_evidence: Optional[str] = None

    sites: Optional[List[SiteInfo]] = None

    human_sample_size: Optional[int] = None
    human_sample_size_evidence: Optional[str] = None

    successful_human_analysis_size: Optional[int] = None
    successful_human_analysis_unit: Optional[str] = None
    successful_human_analysis_evidence: Optional[str] = None

    case_study: Optional[bool] = None
    case_study_evidence: Optional[str] = None

    population_study: Optional[bool] = None
    population_study_evidence: Optional[str] = None

    trauma_studied: Optional[bool] = None
    trauma_studied_evidence: Optional[str] = None

    non_traumatic_pathology_studied: Optional[bool] = None
    non_traumatic_pathology_studied_evidence: Optional[str] = None

    biomolecular_studied: Optional[bool] = None
    biomolecular_methods: Optional[List[str]] = None
    biomolecular_methods_evidence: Optional[str] = None

    research_topics: Optional[List[str]] = None
    research_topics_evidence: Optional[str] = None


# =========================================================
# 2. Load API key
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=api_key)


# =========================================================
# 3. Process one paper
# =========================================================

def process_paper(
    txt_path,
    journal
):

    paper_id = txt_path.stem

    print("\n" + "=" * 70)
    print(f"Processing: {paper_id}")

    # Read current paper
    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    print("Characters read:", len(text))

    print("\nPreview:")
    print(text[:500])


# =========================================================
# 4. Prompt
# =========================================================

    prompt = f"""
You are extracting structured information from a
bioarchaeology or paleopathology publication.

GENERAL RULES

1. Extract information only from the supplied text.
2. Do not infer information that is not explicitly stated.
3. If information cannot be determined, return null.
4. Evidence must be a short excerpt copied from the supplied text.
5. Do not invent evidence.
6. If no suitable textual evidence exists, return null for the
   evidence field.
7. Author affiliations must NOT be treated as excavation institutions
   or archaeological sites.
8. Distinguish the number of human individuals from numbers of graves,
   burials, bones, skeletal elements, lesions, or pathological cases.
9.For bibliographic metadata such as title and authors, copy the text
verbatim whenever possible. Never reconstruct missing information from
external knowledge.

FIELD DEFINITIONS

title:
The title of the publication.
Return null if it cannot be determined.

publication_year:
The year the publication was published.
Return null if it cannot be determined.

authors:
Names of publication authors.
Return a list of names.
Return null if they cannot be determined.

author_affiliations:
Institutional affiliations explicitly associated with the authors.
Return a list.
Return null if they cannot be determined.

study_scope:
Classify the geographic/archaeological scope of the human skeletal sample
directly analysed in the PRESENT study.

Use one of:

"single_site"
Human remains directly analysed in the present study come from one
archaeological site, cemetery, burial ground, or skeletal assemblage.

A study may still be "single_site" even if:
- many individuals are analysed
- individuals are divided into subgroups
- groups are compared by sex, age, tomb size, status, period, or burial type

"multi_site"
Human remains directly analysed in the present study come from two or more
archaeological sites, cemeteries, burial grounds, or skeletal assemblages.

Do NOT count:
- comparative sites represented only by previously published data
- background sites
- sites mentioned only in the Introduction or Discussion
- sites whose human remains were not directly included in the present study

If the number of directly analysed sites cannot be determined, return null.


study_scope_evidence:
Provide a short exact excerpt showing whether the present study directly
analyses human remains from one site or from multiple sites.

The evidence should refer to the actual human skeletal sample used in this
publication, not to comparative or background sites.

sites:
IMPORTANT:
A site MUST be included if the authors explicitly state that human remains
from the site were selected, sampled, submitted, or intended for direct
analysis in the present study, EVEN IF the analysis later failed and produced
zero usable results.

For example, if three sites were selected for analysis but one site produced
no usable collagen, all three sites must be returned. The failed site should
have analysis_successful = false.

Do NOT include:
- sites used only for comparison using previously published data
- sites mentioned only for regional context
- sites mentioned only for chronological context
- sites mentioned only in the Introduction or Discussion
- sites cited from previous publications
- background or literature-review sites
- sites whose published data are reused in comparative or statistical models
  but whose human remains were not directly examined in the present study

The key question is:
Were human remains from this site directly included in the authors'
own research design in this publication?

If no, do not return the site.

For EACH site, return the following fields:

site_name:
The archaeological site, cemetery, burial ground, or assemblage name.

Do not return:
- author institutions
- laboratories
- museums
- general geographic regions unless they are themselves the named site

site_location:
The explicitly stated geographic location of the site.

This may include:
- village or town
- county
- city
- province or autonomous region
- broader geographic region
- country

Examples:
- "Turpan Basin, Xinjiang, China"
- "Xi'an, Shaanxi, China"
- "Liaoning, China"

Use only geographic information explicitly associated with the site
in the supplied text.

Do not infer missing administrative levels.
Do not geocode the site.
Do not add modern locations from external knowledge.

If several levels of location are explicitly stated, combine them into
one concise string from specific to broad where possible.

If the site's location is unclear, return null.

absolute_date:
Any explicit numerical or calendrical date or date range associated
with this specific site.

Examples:
- "1000–700 BCE"
- "250–100 BCE"
- "AD 618–907"
- "ca. 300 BC"
- "7th–9th century CE"

Copy the chronological expression as stated in the text.
Do not convert between BCE/BC or CE/AD.
Do not calculate dates.
Do not infer dates from cultural labels.

If no explicit absolute date is given for that site, return null.

period:
Any explicit archaeological or historical period associated with
this specific site or population.

Examples:
- "Neolithic"
- "Bronze Age"
- "Early Iron Age"
- "Bronze–Early Iron Age"
- "Tang dynasty"
- "Ming dynasty"
- "Qing dynasty"

Do not infer a period solely from numerical dates.
For example, if the text only states "800–400 BCE", do not independently
label it "Iron Age" unless the publication explicitly does so.

If no explicit period is stated, return null.

culture:
Any explicitly stated archaeological culture, cultural tradition,
ethnic designation, polity, or named historical population associated
with the site.

Examples:
- "Subeixi culture"
- "Xiongnu"
- "Scythian"
- "Tang"
- "Uyghur"

Do not infer cultural affiliation from geography or chronology alone.
If unclear or not stated, return null.

For every non-null site field, provide a separate short exact excerpt
supporting that field.

analysis_successful:
Return true if the present study obtained usable analytical data from
human remains from this site.

Return false if human remains from this site were directly included in
the present study and analysis was attempted, but no usable data were obtained.

Examples of unsuccessful analysis may include:
- poor collagen preservation
- insufficient endogenous DNA
- contamination
- analytical failure
- insufficient preservation for the intended analysis

Do NOT use false for comparative or background sites.
Those sites should not be returned in the sites list at all.

If it is unclear whether usable data were obtained, return null.


For every non-null site field, provide a separate short exact excerpt
supporting that field.

analysis_status_evidence:
A short exact excerpt showing whether usable analytical data were
successfully obtained from human remains from this site.

site_name_evidence:
Exact text supporting the site name.

site_location_evidence:
Exact text explicitly supporting the geographic location.
Do not report geographic levels that are not supported by this evidence.

absolute_date_evidence:
Exact text supporting the numerical or calendrical date.

period_evidence:
Exact text explicitly stating the archaeological or historical period.
Do not infer a period from numerical dates.

culture_evidence:
Exact text explicitly supporting the cultural or population attribution.

If no supporting text can be provided for a field, return null for both
that field and its evidence.


The key question is whether human remains from that site are directly
included in the analysis conducted in this publication.

human_sample_size:
The total number of HUMAN INDIVIDUALS directly included in the present study.

This should refer to human individuals, skeletons, or clearly individual-level
human remains included in the authors' own analysis.

A number of complete or partial skeletons may count as human individuals when
the wording clearly refers to separate skeletons representing separate people.

Do NOT use:
- number of animal samples
- number of bones
- number of teeth
- number of crania unless explicitly stated to represent individuals
- number of laboratory measurements
- number of isotope values
- number of successful collagen samples unless they clearly correspond to
  separate human individuals

If unclear, return null.

human_sample_size_evidence:
A short exact excerpt supporting the number of human individuals included.


successful_human_analysis_size:
The number of HUMAN INDIVIDUALS that produced usable results for the
main analysis.

Return a number ONLY when the text explicitly identifies the units as
human individuals, humans, skeletons representing separate individuals,
or otherwise clearly individual-level human cases.

Do NOT use a number if it may include:
- animal samples
- combined human and animal samples
- bones
- collagen samples
- laboratory samples
- measurements
- isotope values
- analytical data points

A generic statement such as "n = 21 samples" or "n = 21 data" must NOT
be interpreted as 21 human individuals unless the text explicitly states
that all 21 are human individuals.

If the total number of successful HUMAN individuals must be obtained by
adding explicitly reported site-specific human counts, this is allowed.

For example:
"one human at Site A" + "humans at Site B (n = 2)"
may be returned as 3 human individuals.

If the successful number of human individuals cannot be determined
unambiguously, return null.

successful_human_analysis_unit:
Use "individuals" or "skeletons" only when the corresponding number
clearly refers to human individuals.

Do not return "samples", "data", "measurements", or other non-individual
units for successful_human_analysis_size.

If the number does not clearly represent human individuals, return null
for both this field and successful_human_analysis_size.

successful_human_analysis_evidence:
A short exact excerpt supporting the number of human individuals with usable
analytical results.

case_study:
Return true when the publication primarily focuses on one individual
or a small number of unusual individuals, rare diseases, unusual
lesions, or an osteobiographical case.

Return false when the paper primarily analyses a population or
assemblage rather than individual cases.

Return null if the study design cannot be determined.

case_study_evidence:
A short exact excerpt supporting the classification where possible.
Do not invent evidence simply to justify false.

population_study:
Return true when the publication systematically analyses a skeletal
population, cemetery population, or assemblage.

Return false when it is primarily an individual case report.

Return null if unclear.

population_study_evidence:
Provide a short exact excerpt demonstrating that the publication
actually analyses a skeletal population, cemetery population, or
assemblage.

Do NOT use general methodological statements about the value of
"population perspectives" or population-level research.

The evidence should describe the actual sample or analysis conducted
in this publication.

trauma_studied:
Return true if trauma, injury, fracture, weapon injury, or other traumatic
lesions are systematically examined as a research topic of the study.

Examples that should return true include:
- cranial trauma
- cranial fractures
- postcranial fractures
- interpersonal violence inferred from traumatic injuries
- weapon-related injuries

If traumatic lesions are a main research focus, return true.

Do not return false when the evidence explicitly states that fractures
or trauma are a main research focus.

Return false only when trauma is not systematically studied.
Return null if unclear.

trauma_studied_evidence:
A short exact excerpt showing that trauma is part of the analysis,
where available.

non_traumatic_pathology_studied:
Return true only when NON-TRAUMATIC pathological conditions are
systematically examined.

Examples include:
- infectious disease
- metabolic disease
- degenerative disease
- dental pathology
- neoplasms
- congenital conditions
- other non-traumatic pathological lesions

Trauma, fractures, weapon injuries, dislocations, and injuries alone
must NOT make this field true.

If the study systematically examines trauma only, return false.

Return null if unclear.

non_traumatic_pathology_studied_evidence:
A short exact excerpt supporting the classification when available.
If the value is false because the paper studies trauma only, evidence
may be null.

biomolecular_studied:
Return true if the study systematically uses biomolecular methods on
human remains.

Examples include:
- ancient DNA
- stable isotope analysis
- proteomics
- palaeoproteomics
- ZooMS when applied to human remains
- pathogen DNA
- other molecular analyses

Return false if no biomolecular method is used.
Return null if unclear.

Return true only if biomolecular analyses were directly performed by the
authors as part of the PRESENT study on human remains.

Do NOT return true when the publication only:
- reuses previously published isotope data
- integrates previously recorded biomolecular data
- cites earlier DNA or isotope studies
- uses published biomolecular results for comparison

If biomolecular data are reused from previous studies but no new biomolecular
analysis is performed in the present publication, return false.


biomolecular_methods:
Return a list of biomolecular methods explicitly used in the study.

Examples:
- "stable carbon isotope analysis"
- "stable nitrogen isotope analysis"
- "strontium isotope analysis"
- "oxygen isotope analysis"
- "ancient DNA"
- "mitochondrial DNA"
- "whole-genome sequencing"
- "proteomics"

Only include biomolecular methods directly performed in the present study.
Do not include methods represented only by previously published or previously
recorded data.



biomolecular_methods_evidence:
A short exact excerpt supporting the biomolecular methods extracted.


research_topics:
Return a list of the main research questions or interpretive topics
explicitly investigated in the study.

Examples include:
- "diet"
- "mobility"
- "migration"
- "geographic origin"
- "kinship"
- "genetic ancestry"
- "population affinity"
- "pathogen infection"
- "disease"
- "trauma"
- "violence"
- "activity"
- "health"
- "lifespan"
- "demography"

Only include topics explicitly investigated in this publication.

Do not infer research topics solely from the method used.

For example:
- carbon and nitrogen isotope analysis does NOT automatically mean "diet"
  unless diet is explicitly investigated.
- strontium or oxygen isotope analysis does NOT automatically mean
  "migration" or "mobility".
- ancient DNA does NOT automatically mean "ancestry".

research_topics_evidence:
A short exact excerpt supporting the extracted research topics.


TEXT TO ANALYSE:

{text}
"""


    # =====================================================
    # Send to Gemini
    # =====================================================

    response = call_gemini_with_retry(prompt)

    # Avoid printing the full model response by default.
    # The structured result is saved locally below for review.
    print("\nModel response received and parsed.")


    # =====================================================
    # Parse
    # =====================================================

    if response.parsed is not None:

        result = response.parsed

        if isinstance(result, ExtractionResult):
            data = result.model_dump()

        else:
            data = result

    else:

        data = json.loads(response.text)


    # =====================================================
    # 7. Save raw JSON
    # =====================================================

    journal_json_dir = (
    RAW_JSON_DIR / journal
)

    journal_json_dir.mkdir(
    parents=True,
    exist_ok=True
    )


    with open(
    journal_json_dir / f"{paper_id}.json",
    "w",
    encoding="utf-8"
    ) as f:

        json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
        )


    # =====================================================
    # 8. Prepare publication-level data
    # =====================================================

    publication_data = data.copy()

    # Remove nested site data from publication-level table
    sites_data = publication_data.pop(
        "sites",
        None
    )

    # Add publication ID
    publication_data["publication_id"] = paper_id
    
    # Add journal
    publication_data["journal"] = journal


    # Convert list fields into strings for CSV
    list_fields = [
        "authors",
        "author_affiliations",
        "biomolecular_methods",
        "research_topics"
    ]

    for field in list_fields:

        if publication_data.get(field):

            publication_data[field] = " | ".join(
                publication_data[field]
            )


    # =====================================================
    # 9. Save / update publication-level CSV
    # =====================================================

    publication_df = pd.DataFrame(
        [publication_data]
    )

    if os.path.exists(PUBLICATIONS_FILE):

        existing_publications = pd.read_csv(
            PUBLICATIONS_FILE
        )

        # Remove old version if paper already exists
        if "journal" not in existing_publications.columns:

    # Existing database was IJO-only
            existing_publications["journal"] = "IJO"


        if "publication_id" in existing_publications.columns:

            existing_publications = (
            existing_publications[
                ~(
                (
                    existing_publications[
                        "publication_id"
                    ].astype(str)
                    == paper_id
                )
                &
                (
                    existing_publications[
                        "journal"
                    ].astype(str)
                    == journal
                )
            )
        ]
    )

        publication_df = pd.concat(
            [
                existing_publications,
                publication_df
            ],
            ignore_index=True
        )


    publication_df.to_csv(
        PUBLICATIONS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # 10. Save / update site-level CSV
    # =====================================================

    # Read existing site table
    if os.path.exists(SITES_FILE):

        existing_sites = pd.read_csv(
            SITES_FILE
        )

        # Compatibility with old IJO-only data
        if "journal" not in existing_sites.columns:
            existing_sites["journal"] = "IJO"

        else:
            existing_sites["journal"] = (
                existing_sites["journal"]
                .fillna("IJO")
                .replace("", "IJO")
            )

        # Remove old records for this exact publication
        if "publication_id" in existing_sites.columns:

            existing_sites = existing_sites[
                ~(
                    (
                        existing_sites["publication_id"]
                        .astype(str)
                        == paper_id
                    )
                    &
                    (
                        existing_sites["journal"]
                        .astype(str)
                        == journal
                    )
                )
            ]

    else:

        # Important: define it even when sites.csv does not exist
        existing_sites = pd.DataFrame()


    # =====================================================
    # Add new site records
    # =====================================================

    if sites_data:

        sites_df = pd.DataFrame(
            sites_data
        )

        sites_df.insert(
            0,
            "publication_id",
            paper_id
        )

        sites_df.insert(
            1,
            "journal",
            journal
        )

        sites_df.insert(
            2,
            "publication_title",
            data.get("title")
        )

        sites_df.insert(
            3,
            "publication_year",
            data.get("publication_year")
        )

        if not existing_sites.empty:

            sites_df = pd.concat(
                [
                    existing_sites,
                    sites_df
                ],
                ignore_index=True
            )

    else:

        # No sites extracted:
        # keep records belonging to other papers
        sites_df = existing_sites


    # Save even if current re-run removes the last old site record
    if not sites_df.empty or os.path.exists(SITES_FILE):

        sites_df.to_csv(
            SITES_FILE,
            index=False,
            encoding="utf-8-sig"
        )


    print(f"Saved: {journal} | {paper_id}")



# =========================================================
# 11. Batch processing
# =========================================================

completed_ids = get_completed_ids()

print(f"Already completed: {len(completed_ids)}")


for i, (
    journal,
    txt_path
) in enumerate(
    txt_files,
    start=1
):

    paper_id = txt_path.stem

    print(
        f"\n[{i}/{len(txt_files)}] "
        f"{journal} | {paper_id}"
    )

    key = (
        journal,
        paper_id
    )

    if key in completed_ids:

        print(
            "Already processed — skipping."
        )

        continue

    try:

        process_paper(
            txt_path,
            journal
        )

        completed_ids.add(
            key
        )

    except Exception as e:

        print(
            "ERROR:",
            type(e).__name__,
            str(e)
        )

        error_row = pd.DataFrame([{
            "journal": journal,
            "publication_id": paper_id,
            "file": str(txt_path),
            "error_type": type(e).__name__,
            "error_message": str(e)
        }])

        if os.path.exists(ERROR_FILE):

            error_row.to_csv(
                ERROR_FILE,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig"
            )

        else:

            error_row.to_csv(
                ERROR_FILE,
                index=False,
                encoding="utf-8-sig"
            )

        continue


print("\nBatch extraction finished.")


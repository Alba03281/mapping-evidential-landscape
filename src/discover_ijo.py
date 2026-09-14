import requests
import pandas as pd
import re
import time


# =========================================================
# SettingsS
# =========================================================

ISSN = "1099-1212"

OUTPUT_ALL = "ijo_all_articles.csv"
OUTPUT_CHINA = "ijo_china_candidates.csv"

# 建议填自己的邮箱，Crossref 推荐使用 polite pool
EMAIL = "YOUR_EMAIL_HERE" 

# =========================================================
# China-related search terms
# =========================================================

china_terms = [
    "china",
    "chinese",

    # major regions / provinces
    "xinjiang",
    "tibet",
    "qinghai",
    "gansu",
    "ningxia",
    "inner mongolia",
    "liaoning",
    "jilin",
    "heilongjiang",
    "hebei",
    "henan",
    "shandong",
    "shanxi",
    "shaanxi",
    "sichuan",
    "yunnan",
    "guizhou",
    "guangxi",
    "guangdong",
    "fujian",
    "zhejiang",
    "jiangsu",
    "anhui",
    "hubei",
    "hunan",
    "jiangxi",

    # common archaeological/historical terms
    "neolithic china",
    "bronze age china",
    "iron age china",

    "han dynasty",
    "tang dynasty",
    "song dynasty",
    "yuan dynasty",
    "ming dynasty",
    "qing dynasty",
    "northern wei",
    "eastern zhou",
    "western zhou",

    "xiongnu",
    "uyghur",
    "uighur"
]


# =========================================================
# Helper functions
# =========================================================

def clean_html(text):
    if not text:
        return ""

    # Crossref abstracts sometimes contain XML / HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_year(item):

    for field in [
        "published-print",
        "published-online",
        "published",
        "issued"
    ]:

        date_info = item.get(field)

        if date_info:
            try:
                return date_info["date-parts"][0][0]
            except Exception:
                pass

    return None


def get_authors(item):

    authors = []

    for author in item.get("author", []):

        given = author.get("given", "")
        family = author.get("family", "")

        name = f"{given} {family}".strip()

        if name:
            authors.append(name)

    return " | ".join(authors)


def is_china_candidate(title, abstract):

    text = f"{title} {abstract}".lower()

    return any(term in text for term in china_terms)


# =========================================================
# Download ALL IJO metadata using cursor pagination
# =========================================================

url = f"https://api.crossref.org/journals/{ISSN}/works"

cursor = "*"
rows_per_request = 200
all_rows = []

print("Downloading IJO metadata from Crossref...")

while True:

    params = {
        "filter": "type:journal-article",
        "rows": rows_per_request,
        "cursor": cursor
    }

    if EMAIL != "YOUR_EMAIL_HERE":
        params["mailto"] = EMAIL

    headers = {
        "User-Agent": "bioarch-literature-miner/0.1"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    # useful if Crossref rejects the request again
    if response.status_code != 200:
        print("Crossref error:")
        print("Status:", response.status_code)
        print(response.text[:1000])

    response.raise_for_status()

    message = response.json()["message"]
    items = message.get("items", [])

    if not items:
        break

    for item in items:

        title_list = item.get("title", [])

        if title_list:
            title = clean_html(title_list[0])
        else:
            title = ""

        abstract = clean_html(
            item.get("abstract", "")
        )

        doi = item.get("DOI", "")
        year = get_year(item)
        authors = get_authors(item)

        candidate = is_china_candidate(
            title,
            abstract
        )

        all_rows.append({
            "doi": doi,
            "publication_year": year,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "china_keyword_candidate": candidate
        })

    print(f"Downloaded {len(all_rows)} records...")

    # If fewer than requested are returned, this is the last page
    if len(items) < rows_per_request:
        break

    next_cursor = message.get("next-cursor")

    if not next_cursor:
        break

    cursor = next_cursor

    time.sleep(0.5)


# =========================================================
# Save all articles
# =========================================================

df = pd.DataFrame(all_rows)

df = df.drop_duplicates(
    subset="doi"
)

df = df.sort_values(
    by=[
        "publication_year",
        "title"
    ],
    ascending=[
        False,
        True
    ]
)

df.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# Save China candidates
# =========================================================

china_df = df[
    df["china_keyword_candidate"] == True
].copy()

china_df.to_csv(
    OUTPUT_CHINA,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# Summary
# =========================================================

print("\nFinished.")
print(f"All IJO articles: {len(df)}")
print(f"China candidates: {len(china_df)}")

print("\nSaved:")
print(f"- {OUTPUT_ALL}")
print(f"- {OUTPUT_CHINA}")

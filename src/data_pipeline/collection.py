import requests, json, time, feedparser, os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://export.arxiv.org/api/query"
OUTPUT_DIR = "raw"

START_YEAR, END_YEAR = 2015, 2025

CATEGORIES = [
    "cs.CL",
    "cs.IR",
    "cs.AI",
    "cs.LG"
]

BATCH_SIZE = 100
TIME_OUT = 60
MAX_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.mount(
    "http://",
    requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=3
    )
)

def build_query(category, start, year, month):
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year

    date_start = f"{year}{month:02d}01"
    date_end = f"{next_year}{next_month:02d}01"

    return {
        "search_query": f"cat:{category} AND submittedDate:[{date_start}0000 TO {date_end}0000]",
        "start": start,
        "max_results": BATCH_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "ascending"
    }

def fetch(params, retry=3):
    for i in range(retry):
        try:
            r = session.get(BASE_URL, params=params, timeout=TIME_OUT)
            if r.status_code == 200:
                return feedparser.parse(r.text)
            time.sleep(5 * (i + 1))
        except:
            if i < retry - 1:
                time.sleep(5 * (i + 1))
    return None

def crawl_month(category, year, month):
    start = 0
    results = []

    while True:
        feed = fetch(build_query(category, start, year, month))
        if feed is None or len(feed.entries) == 0:
            break

        for e in feed.entries:
            results.append({
                "id": e.id,
                "title": e.title.replace("\n", " ").strip(),
                "abstract": e.summary.replace("\n", " ").strip(),
                "authors": [a.name for a in e.authors],
                "categories": [t.term for t in e.tags],
                "primary_category": category,
                "published": e.published,
                "year": year,
                "month": month,
                "pdf": next((l.href for l in e.links if l.type == "application/pdf"), None)
            })

        start += BATCH_SIZE
        time.sleep(1)

    return results

def crawl_year(category, year):
    filename = f"{OUTPUT_DIR}/arxiv_{category}_{year}.jsonl"

    if os.path.exists(filename):
        return

    all_results = []

    with tqdm(desc=f"{category} {year}", unit="paper", leave=True) as pbar:
        for month in range(1, 13):
            month_results = crawl_month(category, year, month)
            if month_results:
                all_results.extend(month_results)
                pbar.update(len(month_results))

    if all_results:
        with open(filename, "w", encoding="utf-8") as out:
            for item in all_results:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    tasks = [
        (category, year)
        for category in CATEGORIES
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(crawl_year, cat, yr) for cat, yr in tasks]
        for _ in as_completed(futures):
            pass

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from html import unescape

BASE = "https://www.47g.org"
API_BASE = f"{BASE}/wp-json/wp/v2"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GovContractAssistant/1.0 (+contact@example.com)"})

# Likely CPTs to try in order:
LIKELY_TYPES = [
    "members", "member", "directory", "companies", "company",
    "organizations", "organization", "team", "teams", "profiles", "profile",
    "posts"  # fallback
]

PER_PAGE = 100
SLEEP = 0.25          # polite delay
FOLLOW_DETAIL = True  # fetch each member page HTML to improve city/state & description

CITY_STATE_RE = re.compile(r"([A-Za-z][A-Za-z\.\-\' ]+),\s*([A-Z]{2})(?:\b|$)")

def strip_html(s: str) -> str:
    if not s:
        return ""
    # quick strip of tags
    txt = re.sub(r"<[^>]+>", " ", s)
    txt = unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()

def first_paragraph_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # prefer a meaningful first <p>
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 20:
            return t
    # fallback: whole text
    return soup.get_text(" ", strip=True)[:400]

def extract_city_state_from_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    text = " ".join(text.split())
    m = CITY_STATE_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip().upper()
    return "", ""

def normalize_state(st: str) -> str:
    st = (st or "").strip().upper()
    if re.fullmatch(r"[A-Z]{2}", st):
        return st
    return ""

def wp_get(url: str, **params):
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def discover_types() -> list[str]:
    found = []
    try:
        types = wp_get(f"{API_BASE}/types")
        for slug in types.keys():
            if slug in LIKELY_TYPES:
                found.append(slug)
    except Exception:
        pass
    # Probe endpoints directly too
    for t in LIKELY_TYPES:
        try:
            _ = wp_get(f"{API_BASE}/{t}", per_page=1)
            if t not in found:
                found.append(t)
        except Exception:
            pass
    # ensure stable fallback last
    if "posts" not in found:
        found.append("posts")
    return found

def fetch_type_items(slug: str) -> list[dict]:
    page = 1
    rows: list[dict] = []
    while True:
        try:
            data = wp_get(f"{API_BASE}/{slug}", per_page=PER_PAGE, page=page, _embed="")  # ask to embed if available
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 404):
                break
            raise
        if not data:
            break
        for it in data:
            rows.append(it)
        page += 1
        time.sleep(SLEEP)
    return rows

def scrape_detail_for_city_state_and_desc(url: str) -> tuple[str, str, str]:
    """
    Fetches member page HTML and tries to improve city/state & description.
    Returns (city, state, better_description_or_empty).
    """
    try:
        html = SESSION.get(url, timeout=30).text
    except Exception:
        return "", "", ""
    soup = BeautifulSoup(html, "html.parser")

    # Try obvious fields first
    # Look for elements containing a location label or pattern
    text = soup.get_text(" ", strip=True)
    city, state = extract_city_state_from_text(text)
    city = clean_text(city)
    # Try a better description: first meaningful paragraph in the main content
    # Guess common WP content containers
    main = soup.select_one("article .entry-content, .entry-content, main, .site-main") or soup
    desc = first_paragraph_from_html(str(main))[:500]
    desc = clean_text(desc)
    return city, state, desc

def pick_description(item: dict) -> str:
    # Prefer ACF description if present
    acf = item.get("acf") or {}
    for key in ["description", "desc", "about", "summary"]:
        if isinstance(acf, dict) and acf.get(key):
            return strip_html(str(acf[key]))[:500]

    # then excerpt, then first para of content
    excerpt = strip_html(item.get("excerpt", {}).get("rendered"))
    if excerpt:
        return excerpt[:500]

    content = item.get("content", {}).get("rendered", "")
    para = first_paragraph_from_html(content)
    if para:
        return para[:500]

    # fallback to title
    title = strip_html(item.get("title", {}).get("rendered"))
    return title

def pick_city_state(item: dict) -> tuple[str, str]:
    acf = item.get("acf") or {}
    # Common ACF keys
    candidate_city_keys = ["city", "location_city", "address_city", "hq_city"]
    candidate_state_keys = ["state", "location_state", "address_state", "hq_state", "st"]

    city = ""
    state = ""
    for k in candidate_city_keys:
        if isinstance(acf, dict) and acf.get(k):
            city = strip_html(str(acf[k])); break
    for k in candidate_state_keys:
        if isinstance(acf, dict) and acf.get(k):
            state = normalize_state(str(acf[k])); break

    # If still empty, try metabox/meta dicts some themes expose
    meta = item.get("meta") or {}
    if not city and isinstance(meta, dict):
        for k in candidate_city_keys:
            if meta.get(k):
                city = strip_html(str(meta[k])); break
    if not state and isinstance(meta, dict):
        for k in candidate_state_keys:
            if meta.get(k):
                state = normalize_state(str(meta[k])); break

    # If still missing, attempt to parse any City, ST in combined strings
    if not (city and state):
        blob = " ".join([
            strip_html(item.get("title", {}).get("rendered")),
            strip_html(item.get("excerpt", {}).get("rendered")),
            strip_html(item.get("content", {}).get("rendered")),
        ])
        c2, s2 = extract_city_state_from_text(blob)
        city = city or c2
        state = state or s2

    return city, state

def scrape() -> pd.DataFrame:
    types = discover_types()
    print("Trying types:", types)

    records = []
    for slug in types:
        try:
            items = fetch_type_items(slug)
            if not items:
                continue
            print(f"Fetched {len(items)} rows from type '{slug}'")

            for it in items:
                name = strip_html(it.get("title", {}).get("rendered"))
                if not name:
                    continue

                desc = pick_description(it)
                city, state = pick_city_state(it)

                # If FOLLOW_DETAIL enabled and city/state still missing, try detail page scrape
                link = it.get("link")
                better_desc = ""
                if FOLLOW_DETAIL and (not city or not state or not desc) and link:
                    c3, s3, better_desc = scrape_detail_for_city_state_and_desc(link)
                    city = city or c3
                    state = state or s3
                    if better_desc and len(better_desc) > len(desc):
                        desc = better_desc

                records.append({
                    "name": name,
                    "description": desc,
                    "city": city,
                    "state": state
                })

        except Exception as e:
            print(f"Skipping type '{slug}' due to error: {e}")
        time.sleep(SLEEP)

    # Dedup by name; keep the longest description if duplicates
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["desc_len"] = df["description"].fillna("").str.len()
    df = df.sort_values("desc_len", ascending=False).drop_duplicates(subset=["name"]).drop(columns=["desc_len"])
    # Normalize blanks
    for c in ["name", "description", "city", "state"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["state"] = df["state"].apply(normalize_state)

    # Final cleanup: remove junk words like "Menu"
    for c in ["name", "description", "city", "state"]:
        df[c] = df[c].fillna("").astype(str).apply(clean_text)

    return df.reset_index(drop=True)

JUNK_WORDS = {"menu", "home", "search", "login", "sign up", "contact"}

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    # normalize whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    # remove known junk tokens if the text is *only* those
    if txt.lower() in JUNK_WORDS:
        return ""
    # also strip leading/trailing junky tokens
    parts = [p for p in txt.split() if p.lower() not in JUNK_WORDS]
    return " ".join(parts).strip()

if __name__ == "__main__":
    df = scrape()
    print(f"Scraped {len(df)} companies")
    # Only keep the four requested columns
    keep = ["name", "description", "city", "state"]
    for k in keep:
        if k not in df.columns:
            df[k] = ""
    out = df[keep]
    out.to_csv("47g_companies.csv", index=False)
    print("Saved to 47g_companies.csv")
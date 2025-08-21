# get_relevant_solicitations.py
# SAM.gov v2 fetchers with key rotation, clear quota/auth errors, and client-side filters.

from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import requests, time, html, re

# Production v2 endpoint (per current docs)
SAM_BASE_URL = "https://api.sam.gov/opportunities/v2/search"
SAM_DESC_URL_V1 = "https://api.sam.gov/prod/opportunities/v1/noticedesc"

# ---- Custom errors for friendly UI messages ----
class SamQuotaError(Exception):
    pass

class SamAuthError(Exception):
    pass

class SamBadRequestError(Exception):
    pass


def _mmddyyyy(d: date) -> str:
    return d.strftime("%m/%d/%Y")

def _window_days_back(days_back: int) -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=max(0, int(days_back)))
    return (_mmddyyyy(start), _mmddyyyy(today))

def _mask_key(k: str) -> str:
    if not k:
        return "(none)"
    return f"...{k[-4:]}"


def _request_sam(params: Dict[str, Any], api_keys: List[str]) -> Dict[str, Any]:
    """
    Try each api_key once. If a key hits quota/auth, rotate to the next.
    If all keys fail, raise a clear aggregated error.
    """
    if not api_keys:
        raise ValueError("No SAM.gov API keys provided.")

    errors: list[str] = []

    for key in api_keys:
        try:
            full_params = dict(params)
            full_params["api_key"] = key
            resp = requests.get(SAM_BASE_URL, params=full_params, timeout=30)

            # Try to extract a short message
            txt = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    txt = (j.get("message") or j.get("error") or "") or ""
            except Exception:
                pass

            if resp.status_code == 429:
                errors.append(f"{_mask_key(key)} → 429 Too Many Requests (quota). {txt}")
                continue

            if resp.status_code in (401, 403):
                # Sometimes returns 403 for quota issues
                if any(s in (txt or "").lower() for s in ["exceeded", "limit", "quota"]):
                    errors.append(f"{_mask_key(key)} → {resp.status_code} quota/limit. {txt}")
                else:
                    errors.append(f"{_mask_key(key)} → {resp.status_code} auth error. {txt}")
                continue

            if resp.status_code == 400:
                raise SamBadRequestError(f"400 Bad Request from SAM.gov. {txt or resp.text}")

            resp.raise_for_status()
            return resp.json() or {}

        except SamBadRequestError:
            raise
        except requests.RequestException as e:
            errors.append(f"{_mask_key(key)} → network error: {e}")
            time.sleep(0.3)
            continue

    if errors:
        if any(("quota" in e.lower() or "limit" in e.lower() or "429" in e) for e in errors):
            raise SamQuotaError("All SAM.gov keys appear rate-limited / out of daily quota.")
        raise SamAuthError("All SAM.gov keys failed (auth/network). Check keys or network.")
    raise SamAuthError("SAM.gov request failed.")


def get_sam_raw_v3(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch raw SAM.gov opportunity records for a date window.
    Only date + limit are sent to the server; other filters can be applied client-side if you want.
    """
    filters = filters or {}
    posted_from, posted_to = _window_days_back(days_back)

    params = {
        "limit": int(limit),
        "postedFrom": posted_from,  # MM/dd/YYYY
        "postedTo": posted_to,      # MM/dd/YYYY
    }

    data = _request_sam(params, api_keys)
    raw_records = data.get("opportunitiesData") or data.get("data") or []
    if not raw_records:
        return []

    # Optional client-side filtering (kept simple)
    def _match(rec: Dict[str, Any]) -> bool:
        nts = filters.get("notice_types") or []
        if nts:
            r_type = str(rec.get("noticeType") or rec.get("type") or "").lower()
            if not r_type or not any(nt.lower() in r_type for nt in nts):
                return False

        kws = [k.strip().lower() for k in (filters.get("keywords_or") or []) if k.strip()]
        if kws:
            title = str(rec.get("title") or "").lower()
            desc = str(rec.get("description") or rec.get("synopsis") or "").lower()
            blob = f"{title} {desc}"
            if not any(k in blob for k in kws):
                return False

        naics_targets = [n for n in (filters.get("naics") or []) if n]
        if naics_targets:
            rec_naics = str(rec.get("naicsCode") or rec.get("naics") or "").strip()
            if not rec_naics or rec_naics not in naics_targets:
                return False

        sas = filters.get("set_asides") or []
        if sas:
            rec_sa = str(rec.get("setAsideCode") or rec.get("setAside") or "").lower()
            if not rec_sa or not any(sa.lower() in rec_sa for sa in sas):
                return False

        due_before = filters.get("due_before")
        if due_before:
            resp = (rec.get("responseDate") or rec.get("closeDate") or "")[:10]
            if resp and resp > str(due_before):
                return False

        return True

    return [r for r in raw_records if _match(r)]


def get_raw_sam_solicitations(limit: int, api_keys: List[str]) -> List[Dict[str, Any]]:
    """Back-compat alias: today's raw records with limit=N (no extra filters)."""
    return get_sam_raw_v3(days_back=0, limit=limit, api_keys=api_keys, filters={})

def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k



SAM_SEARCH_URL_V2 = "https://api.sam.gov/prod/opportunities/v2/search"
SAM_DESC_URL_V1   = "https://api.sam.gov/prod/opportunities/v1/noticedesc"

def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k

def _http_get(url: str, params: dict, key: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "kip_external/1.0"}
    return requests.get(url, params={**params, "api_key": key}, headers=headers, timeout=timeout)

def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    """
    Try to fetch a *single* record using the v2 search endpoint by noticeid.
    Returns {} on failure.
    """
    if not notice_id or not api_keys:
        return {}
    rot = _rotate_keys(api_keys)
    last_exc = None
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_SEARCH_URL_V2, {"noticeid": notice_id, "limit": 1}, key)
            if r.status_code == 429:
                time.sleep(1.0); continue
            r.raise_for_status()
            data = r.json() if r.headers.get("Content-Type","").startswith("application/json") else {}
            # v2 payload typically under "opportunitiesData"
            items = data.get("opportunitiesData") or data.get("data") or []
            if isinstance(items, list) and items:
                return items[0]
            return {}
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue
    return {}

def fetch_notice_description(notice_id: str, api_keys: List[str]) -> str:
    """
    Get full description text using two strategies:
      1) v2 detail (if it includes description/synopsis/long text)
      2) v1 noticedesc (HTML/plain -> normalized text)
    Returns 'None' if both fail.
    """
    # 1) try v2 detail
    detail = fetch_notice_detail_v2(notice_id, api_keys)
    for k in ("description","synopsis","longDescription","fullDescription","additionalInfo"):
        val = detail.get(k)
        if val and str(val).strip():
            return re.sub(r"\s+", " ", str(val)).strip()

    # 2) v1 noticedesc
    rot = _rotate_keys(api_keys)
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_DESC_URL_V1, {"noticeid": notice_id}, key)
            if r.status_code == 429:
                time.sleep(1.0); continue
            r.raise_for_status()
            text = r.text or ""
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)     # strip tags
            text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
            if text:
                return text
        except Exception:
            time.sleep(0.5)
            continue
    return "None"

def _pick_response_date(rec: Dict[str, Any], detail: Dict[str, Any]) -> str:
    """
    SAM may surface the due/response date under several keys & levels.
    We check both the main record and the fetched detail.
    """
    candidates = [
        "responseDate", "responseDueDate", "closeDate", "dueDate", "responseDateTime",
        # sometimes nested under 'dates' or similar
    ]
    for src in (rec, detail):
        for k in candidates:
            val = src.get(k)
            if val not in (None, "", []):
                return str(val)
        # shallow nested maps
        for v in src.values():
            if isinstance(v, dict):
                for k in candidates:
                    if v.get(k):
                        return str(v[k])
    return "None"

def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, dict):
            # pick common subkeys if dict
            for sub in ("name","value","code","text"):
                if v.get(sub):
                    return str(v[sub])
            continue
        if v is not None and str(v).strip() != "":
            return str(v)
    return default

def map_record_allowed_fields(
    rec: Dict[str, Any],
    *,
    api_keys: Optional[List[str]] = None,
    fetch_desc: bool = True
) -> Dict[str, Any]:
    notice_id            = _first_nonempty(rec, "noticeId", "id")
    solicitation_number  = _first_nonempty(rec, "solicitationNumber", "solicitationNo")
    title                = _first_nonempty(rec, "title")
    notice_type          = _first_nonempty(rec, "noticeType", "type")
    posted_date          = _first_nonempty(rec, "postedDate", "publicationDate")
    archive_date         = _first_nonempty(rec, "archiveDate")
    naics_code           = _first_nonempty(rec, "naicsCode", "naics")
    set_aside_code       = _first_nonempty(rec, "setAsideCode", "typeOfSetAside", "setAside")

    # link (handy reference)
    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = _first_nonempty(rec, "url", "samLink")

    # detail & description
    detail = {}
    if fetch_desc and api_keys and notice_id != "None":
        detail = fetch_notice_detail_v2(notice_id, api_keys)

    # response_date from rec or detail
    response_date = _pick_response_date(rec, detail)

    # description from inline/rec first
    description = _first_nonempty(rec, "description", "synopsis", default="")
    def _looks_like_placeholder_or_url(t: str) -> bool:
        if not t: return True
        s = t.strip().lower()
        if s in ("none","n/a","na"): return True
        return s.startswith("http://") or s.startswith("https://")

    if _looks_like_placeholder_or_url(description):
        if fetch_desc and api_keys and notice_id != "None":
            description = fetch_notice_description(notice_id, api_keys)
        else:
            description = "None"

    return {
        "notice_id":            notice_id,
        "solicitation_number":  solicitation_number,
        "title":                title,
        "notice_type":          notice_type,
        "posted_date":          posted_date,
        "response_date":        response_date,
        "archive_date":         archive_date,
        "naics_code":           naics_code,
        "set_aside_code":       set_aside_code,
        "description":          description,
        "link":                 link,
    }

def _deep_get(obj: Any, candidates: List[str]) -> Optional[str]:
    """
    Try to find a value for any of the candidate keys, including if nested under dicts.
    Returns the first non-empty string found.
    """
    if not isinstance(obj, dict):
        return None
    # direct
    for k in candidates:
        v = obj.get(k)
        if v not in (None, "", []):
            return str(v)
    # shallow nested search
    for v in obj.values():
        if isinstance(v, dict):
            found = _deep_get(v, candidates)
            if found:
                return found
    return None
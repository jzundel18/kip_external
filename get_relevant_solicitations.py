# get_relevant_solicitations.py
# Pulls raw SAM.gov opportunities (all fields) and provides a light mapping helper.

from __future__ import annotations
import requests
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import time

# Per the current SAM.gov v2 docs, production URL omits '/prod'
SAM_BASE_URL = "https://api.sam.gov/opportunities/v2/search"

# ---- Custom errors so the UI can show a friendly message ----
class SamQuotaError(Exception):
    def __init__(self, key_label: str, message: str, status: int | None = None):
        super().__init__(message)
        self.key_label = key_label
        self.status = status

class SamAuthError(Exception):
    def __init__(self, key_label: str, message: str, status: int | None = None):
        super().__init__(message)
        self.key_label = key_label
        self.status = status

class SamBadRequestError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

def _mmddyyyy(d: date) -> str:
    return d.strftime("%m/%d/%Y")

def _window_days_back(days_back: int) -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=max(0, int(days_back)))
    return (_mmddyyyy(start), _mmddyyyy(today))

def _mask_key(k: str) -> str:
    if not k:
        return "(none)"
    return f"...{k[-4:]}"  # show only last 4

def _pick_api_key(api_keys: List[str], attempt: int) -> Optional[str]:
    if not api_keys:
        return None
    return api_keys[attempt % len(api_keys)]

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

            # Quick, readable reason if SAM returns JSON with message
            txt = ""
            try:
                j = resp.json()
                txt = (j.get("message") or j.get("error") or "") if isinstance(j, dict) else ""
            except Exception:
                pass

            if resp.status_code == 429:
                errors.append(f"{_mask_key(key)} → 429 Too Many Requests (quota). {txt}")
                # try next key
                continue

            if resp.status_code in (401, 403):
                # Distinguish quota text if present in body
                if any(s in (txt or "").lower() for s in ["exceeded", "limit", "quota"]):
                    errors.append(f"{_mask_key(key)} → {resp.status_code} quota/limit. {txt}")
                else:
                    errors.append(f"{_mask_key(key)} → {resp.status_code} auth error. {txt}")
                continue

            if resp.status_code == 400:
                # Bad params (dates, etc) – don't rotate, this is caller-side
                raise SamBadRequestError(f"400 Bad Request from SAM.gov. {txt or resp.text}", status=400)

            resp.raise_for_status()
            return resp.json() or {}

        except SamBadRequestError:
            # Propagate immediately: caller should fix inputs
            raise
        except requests.RequestException as e:
            errors.append(f"{_mask_key(key)} → network error: {e}")
            # try next key
            time.sleep(0.3)
            continue

    # All keys exhausted
    if errors:
        # If any mention of quota/limit, raise a quota error for clarity
        if any(("quota" in e.lower() or "limit" in e.lower() or "429" in e) for e in errors):
            raise SamQuotaError("all-keys", "All SAM.gov keys appear rate-limited / out of daily quota.", None)
        raise SamAuthError("all-keys", "All SAM.gov keys failed (auth/network). Check keys or network.", None)

    # Fallback (shouldn't hit)
    raise SamAuthError("all-keys", "SAM.gov request failed.", None)
def get_sam_raw_v3(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch **raw SAM.gov opportunity records** (all fields) for a date window.

    Args:
        days_back: 0 = only today; N = from (today - N) to today inclusive
        limit: max results to request (SAM caps per-request; we do a single call)
        api_keys: list of SAM.gov API keys; we rotate if a key fails
        filters: optional dict; server sends only date-range & limit, we apply the rest client-side:
            - "notice_types": List[str]  (e.g., ["Solicitation","Combined Synopsis/Solicitation"])
            - "keywords_or": List[str]   (lowercased OR-match against title+description)
            - "naics": List[str]         (strings, e.g., ["541511","541512"])
            - "set_asides": List[str]    (strings)
            - "agency_contains": str     (substring, case-insensitive)
            - "due_before": str          (YYYY-MM-DD)

    Returns:
        List of dicts — each dict is the **entire SAM.gov record** (no fields lost).
    """
    filters = filters or {}
    posted_from, posted_to = _window_days_back(days_back)

    params = {
        "limit": int(limit),
        "postedFrom": posted_from,  # MM/dd/YYYY
        "postedTo": posted_to,      # MM/dd/YYYY
        # You may optionally add server-side filters here if desired:
        # "ptype": "o,k,r,s",         # procurement types (comma-separated codes)
        # "ncode": "541511",          # NAICS
        # "typeOfSetAside": "SBA",    # one code at a time
        # "organizationName": "Defense Logistics Agency",
        # "rdlfrom": "08/01/2025",
        # "rdlto":   "08/31/2025",
    }
    # Remove empty params
    params = {k: v for k, v in params.items() if v not in (None, "", [], {})}

    data = _request_sam(params, api_keys)
    raw_records = data.get("opportunitiesData") or data.get("data") or []
    if not raw_records:
        return []

    # Client-side filtering (non-destructive) to keep it simple & robust
    def _match(rec: Dict[str, Any]) -> bool:
        # notice types (OR contains match)
        nts = filters.get("notice_types") or []
        if nts:
            r_type = str(rec.get("noticeType") or rec.get("type") or "").lower()
            if not r_type or not any(nt.lower() in r_type for nt in nts):
                return False

        # keywords OR on title + description
        kws = [k.strip().lower() for k in (filters.get("keywords_or") or []) if k.strip()]
        if kws:
            title = str(rec.get("title") or "").lower()
            desc = str(rec.get("description") or rec.get("synopsis") or "").lower()
            blob = f"{title} {desc}"
            if not any(k in blob for k in kws):
                return False

        # NAICS exact
        naics_targets = [n for n in (filters.get("naics") or []) if n]
        if naics_targets:
            rec_naics = str(rec.get("naics") or rec.get("naicsCode") or "").strip()
            if not rec_naics or rec_naics not in naics_targets:
                return False

        # set-aside
        sas = filters.get("set_asides") or []
        if sas:
            rec_sa = str(rec.get("setAside") or rec.get("setAsideCode") or "").lower()
            if not rec_sa or not any(sa.lower() in rec_sa for sa in sas):
                return False

        # agency contains
        agency_sub = (filters.get("agency_contains") or "").strip().lower()
        if agency_sub:
            agency = str(rec.get("department") or rec.get("agency") or rec.get("organizationName") or "").lower()
            if agency_sub not in agency:
                return False

        # due before (compare as text on YYYY-MM-DD prefix)
        due_before = filters.get("due_before")
        if due_before:
            resp = (rec.get("responseDate") or rec.get("closeDate") or "")[:10]
            if resp and resp > str(due_before):
                return False

        return True

    return [r for r in raw_records if _match(r)]

# Back-compat alias (today only)
def get_raw_sam_solicitations(limit: int, api_keys: List[str]) -> List[Dict[str, Any]]:
    return get_sam_raw_v3(days_back=0, limit=limit, api_keys=api_keys, filters={})

# Minimal mapping for old table UI (kept for compatibility)
_TABLE_HEADER = [
    "notice id",
    "notice type",
    "solicitation number",
    "title",
    "posted date",
    "due date",
    "NAICS Code",
    "set-aside",
    "agency",
    "solicitation link",
    "item description",
]

def _map_record_to_row(rec: Dict[str, Any]) -> List[str]:
    def g(*keys: str, default: str = "") -> str:
        for k in keys:
            v = rec.get(k)
            if v is not None:
                return str(v)
        return default

    return [
        g("noticeId", "id"),
        g("noticeType", "type"),
        g("solicitationNumber", "solnum"),
        g("title"),
        g("postedDate", "publishDate"),
        g("responseDate", "closeDate"),
        g("naics", "naicsCode"),
        g("setAside", "setAsideCode"),
        g("department", "agency", "organizationName"),
        g("url", "samLink"),
        g("description", "synopsis"),
    ]

def get_relevant_solicitations_v2(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Dict[str, Any],
    openai_api_key: Optional[str] = None,
) -> List[List[str]]:
    raw = get_sam_raw_v3(days_back=days_back, limit=limit, api_keys=api_keys, filters=filters)
    rows = [_map_record_to_row(r) for r in raw]
    return [_TABLE_HEADER, *rows]
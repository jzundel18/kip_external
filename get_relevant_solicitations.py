# get_relevant_solicitations.py
# Pulls raw SAM.gov opportunities (all fields) and provides a light mapping helper.

from __future__ import annotations
import requests
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import itertools
import time


SAM_BASE_URL = "https://api.sam.gov/prod/opportunities/v2/search"


def _today_iso() -> str:
    """YYYY-MM-DD (SAM.gov expects date-only for postedFrom/postedTo)."""
    return date.today().isoformat()


def _iso_days_back(days_back: int) -> tuple[str, str]:
    """
    Return (postedFrom, postedTo) as YYYY-MM-DD.
    days_back = 0 => today only
    days_back = N => [today - N, today]
    """
    today = date.today()
    start = today - timedelta(days=days_back)
    return (start.isoformat(), today.isoformat())


def _pick_api_key(api_keys: List[str], attempt: int) -> Optional[str]:
    if not api_keys:
        return None
    return api_keys[attempt % len(api_keys)]


def _request_sam(params: Dict[str, Any], api_keys: List[str], max_attempts: int = 3) -> Dict[str, Any]:
    """
    Make a single SAM.gov request with basic retry & key rotation.
    Raises the last error if all attempts fail.
    """
    last_exc = None
    for attempt in range(max_attempts):
        key = _pick_api_key(api_keys, attempt)
        if not key:
            raise ValueError("No SAM.gov API key provided.")
        try:
            full_params = dict(params)
            full_params["api_key"] = key
            resp = requests.get(SAM_BASE_URL, params=full_params, timeout=30)
            # If SAM returns non-2xx, raise for_status to capture the body
            resp.raise_for_status()
            return resp.json() or {}
        except requests.HTTPError as e:
            # If quota or auth issue, rotate key on next attempt
            last_exc = e
            time.sleep(0.5)
            continue
        except requests.RequestException as e:
            last_exc = e
            time.sleep(0.5)
            continue
    # All attempts failed
    if last_exc:
        raise last_exc
    return {}


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
        limit: max results to request (SAM may cap per-request; we do a single call)
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
    posted_from, posted_to = _iso_days_back(max(0, int(days_back)))

    # Only send minimal filters to SAM itself (date window + limit).
    # You can pass "noticeType" to SAM, but it’s safer to filter client-side
    # to avoid accidental rejections until your needs are clear.
    params = {
        "limit": int(limit),
        "postedFrom": posted_from,
        "postedTo": posted_to,
        # "noticeType": ",".join(filters.get("notice_types", [])) if filters.get("notice_types") else None,
        # (Uncomment above to filter on server by notice type if you prefer)
    }
    # Remove Nones so they don't appear as "None" in the query string
    params = {k: v for k, v in params.items() if v not in (None, "", [])}

    data = _request_sam(params, api_keys)
    raw_records = data.get("opportunitiesData") or data.get("data") or []
    if not raw_records:
        return []

    # Client-side filtering (non-destructive): produce a filtered view
    def _match(rec: Dict[str, Any]) -> bool:
        # notice types (OR contains match)
        nts = filters.get("notice_types") or []
        if nts:
            # records sometimes use 'type' or 'noticeType'
            r_type = str(rec.get("noticeType") or rec.get("type") or "").lower()
            if r_type:
                if not any(nt.lower() in r_type for nt in nts):
                    return False
            else:
                return False

        # keywords OR on title + description
        kws = [k.strip().lower() for k in (filters.get("keywords_or") or []) if k.strip()]
        if kws:
            title = str(rec.get("title") or "").lower()
            desc = str(rec.get("description") or rec.get("synopsis") or "").lower()
            blob = f"{title} {desc}"
            if not any(k in blob for k in kws):
                return False

        # NAICS (exact string match on common keys)
        naics_targets = [n for n in (filters.get("naics") or []) if n]
        if naics_targets:
            rec_naics = str(rec.get("naics") or rec.get("naicsCode") or "").strip()
            if rec_naics and rec_naics not in naics_targets:
                return False
            if not rec_naics:
                return False

        # set-aside (contains match)
        sas = filters.get("set_asides") or []
        if sas:
            rec_sa = str(rec.get("setAside") or rec.get("setAsideCode") or "").lower()
            if rec_sa:
                if not any(sa.lower() in rec_sa for sa in sas):
                    return False
            else:
                return False

        # agency contains
        agency_sub = (filters.get("agency_contains") or "").strip().lower()
        if agency_sub:
            agency = str(rec.get("department") or rec.get("agency") or rec.get("organizationName") or "").lower()
            if agency_sub not in agency:
                return False

        # due before
        due_before = filters.get("due_before")
        if due_before:
            # rec may use responseDate or closeDate; compare as plain dates if possible
            resp = (rec.get("responseDate") or rec.get("closeDate") or "")[:10]  # YYYY-MM-DD*
            if resp and resp > str(due_before):
                return False

        return True

    filtered = [r for r in raw_records if _match(r)]
    return filtered


# Back-compat alias for your app code if you call this name somewhere else
def get_raw_sam_solicitations(limit: int, api_keys: List[str]) -> List[Dict[str, Any]]:
    """Alias: today's raw records with limit=N (no extra filters)."""
    return get_sam_raw_v3(days_back=0, limit=limit, api_keys=api_keys, filters={})


# --------- Optional: minimal table mapping for old UI paths ---------

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
    """Map a raw SAM record to the minimal table row used by the UI."""
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
    """
    Return a 2D list: [header_row, *data_rows], preserving your app's existing expectations.
    Under the hood we fetch raw→filter client-side→map to the minimal table.

    NOTE: We do not call OpenAI here; AI downselect can be layered in your app if desired.
    """
    raw = get_sam_raw_v3(days_back=days_back, limit=limit, api_keys=api_keys, filters=filters)
    rows = [_map_record_to_row(r) for r in raw]
    return [_TABLE_HEADER, *rows]
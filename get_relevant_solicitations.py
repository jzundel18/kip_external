# get_relevant_solicitations.py
# SAM.gov v2 fetchers with key rotation, clear quota/auth errors, and client-side filters.

from __future__ import annotations
import requests
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import time

# Production v2 endpoint (per current docs)
SAM_BASE_URL = "https://api.sam.gov/opportunities/v2/search"

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

        agency_sub = (filters.get("agency_contains") or "").strip().lower()
        if agency_sub:
            agency = str(rec.get("department") or rec.get("agency") or rec.get("organizationName") or "").lower()
            if agency_sub not in agency:
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


# Helper: map ONE raw record -> ONLY the allowed fields you want to persist
def map_record_allowed_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    def g(*keys: str, default: str = "") -> str:
        for k in keys:
            v = rec.get(k)
            if v is not None and v != "":
                return str(v)
        return default

    # Link: prefer links[0].href if present
    link = ""
    links = rec.get("links")
    if isinstance(links, list) and links:
        first = links[0]
        if isinstance(first, dict):
            link = str(first.get("href") or "")
    if not link:
        link = g("url", "samLink", default="")

    # Place of performance
    pop = rec.get("placeOfPerformance") or {}
    place_city = ""
    place_state = ""
    place_country_code = ""
    if isinstance(pop, dict):
        place_city = str(pop.get("city") or "")
        place_state = str(pop.get("state") or "")
        place_country_code = str(pop.get("countryCode") or "")

    return {
        "notice_id":            g("noticeId", "id"),
        "solicitation_number":  g("solicitationNumber"),
        "title":                g("title"),
        "notice_type":          g("noticeType", "type"),
        "posted_date":          g("postedDate"),
        "response_date":        g("responseDate", "closeDate"),
        "archive_date":         g("archiveDate"),
        "department":           g("department"),
        "agency":               g("agency"),
        "office":               g("office"),
        "organization_name":    g("organizationName"),
        "naics_code":           g("naicsCode", "naics"),
        "naics_description":    g("naicsDescription"),
        "classification_code":  g("classificationCode"),
        "set_aside_code":       g("setAsideCode", "setAside"),
        "set_aside_description":g("setAsideDescription"),
        "description":          g("description", "synopsis"),
        "link":                 link,
        "place_city":           place_city,
        "place_state":          place_state,
        "place_country_code":   place_country_code,
    }
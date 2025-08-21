# get_relevant_solicitations.py
# SAM.gov v2 fetchers with key rotation, clear quota/auth errors, and client-side filters.

from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import requests, time, html, re


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

def map_record_allowed_fields(rec: Dict[str, Any], *, api_keys: list[str] | None = None, fetch_desc: bool = True) -> Dict[str, Any]:
    """
    Map a SAM /search record into the exact fields you persist.
    If fetch_desc=True and api_keys provided, fetch full description via noticedesc.
    When a field is not present, return the string 'None' (per your request).
    """
    def first_nonempty(*keys, default="None"):
        for k in keys:
            v = rec.get(k)
            # some fields can be nested dicts (e.g., organization)
            if isinstance(v, dict):
                # common nested possibilities
                for subk in ("name", "value", "code"):
                    if v.get(subk):
                        return str(v[subk])
                continue
            if v is not None and str(v).strip() != "":
                return str(v)
        return default

    # response/close date—SAM sometimes uses responseDate or closeDate
    response_date = first_nonempty("responseDate", "closeDate")
    posted_date   = first_nonempty("postedDate", "publicationDate")
    archive_date  = first_nonempty("archiveDate")

    # organization/agency—SAM records vary; try multiple common paths
    agency = first_nonempty("agency", "organization", "department")
    organization_name = first_nonempty("organizationName", "organization", "office")

    # set-aside—SAM may use setAsideCode, typeOfSetAside, or setAside
    set_aside_code = first_nonempty("setAsideCode", "typeOfSetAside", "setAside")

    # link—if provided in links[0].href, else url/samLink
    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = first_nonempty("url", "samLink")

    # NAICS can appear as a simple string or nested
    naics_code = first_nonempty("naicsCode", "naics")

    # Title/notice type/solicitation number
    title = first_nonempty("title")
    notice_type = first_nonempty("noticeType", "type")
    solicitation_number = first_nonempty("solicitationNumber", "solicitationNo")

    notice_id = first_nonempty("noticeId", "id")

    # Description:
    # 1) Prefer a description/synopsis field if present
    # 2) Otherwise fetch via noticedesc endpoint
    description = first_nonempty("description", "synopsis", default="")
    if (description == "" or description.lower().startswith("http")) and fetch_desc and api_keys and notice_id != "None":
        fetched = fetch_notice_description(notice_id, api_keys)
        description = fetched or "None"
    elif description == "":
        description = "None"

    return {
        "notice_id":            notice_id,
        "solicitation_number":  solicitation_number,
        "title":                title,
        "notice_type":          notice_type,
        "posted_date":          posted_date,
        "response_date":        response_date,
        "archive_date":         archive_date,
        "agency":               agency,
        "organization_name":    organization_name,
        "naics_code":           naics_code,
        "set_aside_code":       set_aside_code,
        "description":          description,
        "link":                 link,
    }

def _rotate_keys(keys):
    for k in keys:
        yield k

def fetch_notice_description(notice_id: str, api_keys: list[str], timeout=20) -> str:
    """
    Fetch the actual solicitation description from the SAM 'noticedesc' endpoint.
    Returns plain text. Falls back to '' if unavailable.
    """
    if not notice_id or not api_keys:
        return ""

    round_keys = _rotate_keys(api_keys)
    last_exc = None
    for _ in range(len(api_keys)):
        key = next(round_keys)
        try:
            resp = requests.get(
                SAM_DESC_URL,
                params={"noticeid": notice_id, "api_key": key},
                timeout=timeout,
            )
            if resp.status_code == 429:
                time.sleep(1.0)
                continue
            resp.raise_for_status()

            # SAM noticedesc often returns raw HTML or plain text; normalize to text
            text = resp.text or ""
            # Unescape HTML entities
            text = html.unescape(text)
            # Strip basic HTML tags (lightweight)
            text = re.sub(r"<[^>]+>", " ", text)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue
    # If all keys failed:
    return ""

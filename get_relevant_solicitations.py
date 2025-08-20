# get_relevant_solicitations.py
#
# UI-driven fetch of SAM.gov opportunities.
# - Pushes filters from the UI directly into the SAM.gov API query.
# - Optional AI downselect based on a short company description.
# - Returns list-of-lists: [header, *rows] to match your app's existing expectations.
#
# Dependencies: requests, pandas, openai (>=1.0), python-dateutil (for date parsing if needed)

from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from openai import OpenAI


# ------------------------------
# Public entry points
# ------------------------------

def get_relevant_solicitations_v2(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Dict[str, Any],
    openai_api_key: str
) -> List[List[Any]]:
    """
    UI-driven fetch:
      - Pulls from SAM.gov once using the provided filters (no redundant client-side filtering).
      - Optional AI downselect if filters['use_ai_downselect'] is True.
    Returns:
      List-of-lists where first element is header, followed by data rows.
    """
    if not api_keys:
        return []

    # Build base params from UI filters
    params = _build_sam_params(days_back, limit, filters)

    # Fetch (try keys in order; stop after first success)
    records, last_err = _fetch_records_with_key_rotation(params, api_keys)
    if records is None:
        # surface the last error to the caller
        raise last_err if last_err else RuntimeError("Failed to fetch SAM.gov records.")

    # Convert SAM response records to your table (list-of-lists)
    header, rows = _records_to_table(records)

    # Optional AI downselect
    if filters.get("use_ai_downselect") and filters.get("company_desc"):
        rows = _ai_downselect_rows(
            header=header,
            rows=rows,
            company_desc=str(filters.get("company_desc") or ""),
            openai_api_key=openai_api_key
        )

    return [header] + rows


# ---- Backward-compat wrapper (keeps older app code working if anything still calls this) ----
def get_relevant_solicitation_list(
    Days_back: int,
    N_SAM_results: int,
    Api_Keys: List[str],
    target_keywords: List[str],
    OpenAi_API_Key: str
) -> List[List[Any]]:
    """
    Legacy signature used by older UI code.
    Maps to v2 with keywords only. No AI downselect here.
    """
    filters = {
        "keywords_or": target_keywords or [],
        "naics": [],
        "set_asides": [],
        "agency_contains": "",
        "due_before": None,
        "notice_types": [],
        "use_ai_downselect": False,
        "company_desc": "",
    }
    return get_relevant_solicitations_v2(
        days_back=int(Days_back),
        limit=int(N_SAM_results),
        api_keys=Api_Keys or [],
        filters=filters,
        openai_api_key=OpenAi_API_Key or ""
    )


# ------------------------------
# SAM.gov query + conversion
# ------------------------------

def _build_sam_params(days_back: int, limit: int, filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate UI filters to SAM.gov search parameters (Opportunities v2).
    NOTE: SAM parameters can vary by endpoint/version. These names are commonly supported.
    Adjust the names to match the exact endpoint you’re using if needed.
    """
    posted_to = datetime.now(timezone.utc).date()
    posted_from = posted_to - timedelta(days=days_back)

    p = {
        # Date range filter (MM/DD/YYYY)
        "postedFrom": posted_from.strftime("%m/%d/%Y"),
        "postedTo": posted_to.strftime("%m/%d/%Y"),
        # Sorting & pagination
        "sort": "-modifiedDate",
        "limit": max(1, min(int(limit), 200)),
        "offset": 0,
    }

    # Keywords OR (space-separated generally acts like OR in SAM's 'q')
    kws = [k for k in (filters.get("keywords_or") or []) if k]
    if kws:
        p["q"] = " ".join(kws)

    # NAICS (comma-separated list)
    naics = filters.get("naics") or []
    if naics:
        p["naics"] = ",".join([_digits_only(x) for x in naics if x])

    # Set-aside codes (examples: SB, WOSB, EDWOSB, HUBZone, SDVOSB, 8A, SDB)
    sas = filters.get("set_asides") or []
    if sas:
        p["setAsideCode"] = ",".join(sas)

    # Notice types (examples differ—map your UI choices to API values if necessary)
    nts = filters.get("notice_types") or []
    if nts:
        p["noticeType"] = ",".join(nts)

    # Agency/department (broad substring — not perfect; adjust/remove if your endpoint doesn't support)
    agency_contains = (filters.get("agency_contains") or "").strip()
    if agency_contains:
        p["department"] = agency_contains

    # Due date upper bound
    due_before = filters.get("due_before")
    if due_before:
        # Expecting ISO date string "YYYY-MM-DD" from UI
        try:
            dt = datetime.fromisoformat(str(due_before))
            p["responseDateTo"] = dt.strftime("%m/%d/%Y")
        except Exception:
            pass

    return p


def _fetch_records_with_key_rotation(params: Dict[str, Any], api_keys: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Exception]]:
    """
    Try each SAM API key until one works (basic rotation / fallback).
    Returns (records, last_exception).
    """
    base = "https://api.sam.gov/opportunities/v2/search"
    headers = {"Accept": "application/json"}

    last_err: Optional[Exception] = None
    for key in api_keys:
        try:
            p = dict(params)
            p["api_key"] = key
            r = requests.get(base, params=p, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            # Different responses sometimes store the list differently
            recs = data.get("opportunitiesData") or data.get("data") or []
            return recs, None
        except Exception as e:
            last_err = e
            continue

    return None, last_err


def _records_to_table(records: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:
    """
    Convert raw SAM.gov records into your app’s table shape (list-of-lists).
    We keep column names consistent with what your CSV likely expects.
    If a field doesn't exist, we leave it blank.
    """
    header = [
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
        # keep space for fields your downstream code might expect (safe to leave blank)
        "submission instructions",
        "fulfillment instructions",
        "quantity",
        "zip code",
        "listed supplier",
        "item part#",
        "poc name",
        "poc email",
        "poc phone",
    ]

    rows: List[List[Any]] = []
    for r in records:
        # Safely pull common fields; fallback to empty string if not present
        rows.append([
            r.get("noticeId") or r.get("id") or "",
            r.get("noticeType") or r.get("type") or "",
            r.get("solicitationNumber") or r.get("solnum") or "",
            r.get("title") or "",
            r.get("postedDate") or r.get("publishDate") or "",
            r.get("responseDate") or r.get("closeDate") or "",
            r.get("naics") or r.get("naicsCode") or "",
            r.get("setAside") or r.get("setAsideCode") or "",
            r.get("department") or r.get("agency") or r.get("organizationName") or "",
            r.get("url") or r.get("samLink") or "",
            r.get("description") or r.get("synopsis") or "",
            # The remaining are placeholders; fill if/when your endpoint includes them or you derive them
            r.get("submissionInstructions") or "",
            r.get("fulfillmentInstructions") or "",
            r.get("quantity") or "",
            r.get("zip") or r.get("zipcode") or "",
            r.get("listedSupplier") or "",
            r.get("partNumber") or "",
            _first_poc_name(r) or "",
            _first_poc_email(r) or "",
            _first_poc_phone(r) or "",
        ])

    return header, rows


# ------------------------------
# Optional AI downselect
# ------------------------------

def _ai_downselect_rows(
    header: List[str],
    rows: List[List[Any]],
    company_desc: str,
    openai_api_key: str
) -> List[List[Any]]:
    """
    Row-wise binary classification: keep vs drop based on short company description.
    Fast + cheap: uses gpt-4o-mini by default. Adjust model if desired.
    """
    if not rows or not company_desc.strip():
        return rows

    client = OpenAI(api_key=openai_api_key)
    # Precompute index map for quick access
    idx = {col: i for i, col in enumerate(header)}
    i_title = idx.get("title", 0)
    i_desc = idx.get("item description", 0)

    kept: List[List[Any]] = []
    for row in rows:
        title = str(row[i_title] if i_title < len(row) else "")
        desc = str(row[i_desc] if i_desc < len(row) else "")

        prompt = f"""
You are helping downselect government solicitations for a company.

Company description:
\"\"\"{company_desc.strip()}\"\"\"

Opportunity:
Title: {title}
Description: {desc}

Question: Is this opportunity a good fit for the company given the description?
Respond with a strict JSON object on one line: {{"keep": true/false}}
        """.strip()

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.S)
            keep = True
            if m:
                data = json.loads(m.group(0))
                keep = bool(data.get("keep", True))
            if keep:
                kept.append(row)
        except Exception:
            # Fail-open (keep) on any model/network error
            kept.append(row)

    return kept


# ------------------------------
# Small helpers
# ------------------------------

def _digits_only(s: str) -> str:
    return re.sub(r"[^\d]", "", str(s or ""))

def _first_poc_name(rec: Dict[str, Any]) -> str:
    # Try common PoC shapes; adjust if your endpoint differs
    # Some records include a list like "pointOfContact": [{"name":"...","email":"..."}]
    poc = rec.get("pointOfContact") or rec.get("poc") or rec.get("contacts")
    if isinstance(poc, list) and poc:
        name = poc[0].get("name") or poc[0].get("fullName")
        return name or ""
    if isinstance(poc, dict):
        return poc.get("name") or poc.get("fullName") or ""
    return ""

def _first_poc_email(rec: Dict[str, Any]) -> str:
    poc = rec.get("pointOfContact") or rec.get("poc") or rec.get("contacts")
    if isinstance(poc, list) and poc:
        return poc[0].get("email") or ""
    if isinstance(poc, dict):
        return poc.get("email") or ""
    return ""

def _first_poc_phone(rec: Dict[str, Any]) -> str:
    poc = rec.get("pointOfContact") or rec.get("poc") or rec.get("contacts")
    if isinstance(poc, list) and poc:
        return poc[0].get("phone") or poc[0].get("telephone") or ""
    if isinstance(poc, dict):
        return poc.get("phone") or poc.get("telephone") or ""
    return ""

# ---- NEW: raw fetch that returns every field SAM.gov gives us ----
def get_sam_raw_v3(days_back: int, limit: int, api_keys: list[str], filters: dict) -> list[dict]:
    """
    Returns the raw list of opportunity records from SAM.gov (all fields).
    No OpenAI, no downselection. Just raw JSON dicts as the API returns them.
    """
    if not api_keys:
        return []
    params = _build_sam_params(days_back, limit, filters)
    records = []
    last_err = None
    for key in api_keys:
        try:
            data = _sam_search(key, params)
            records = data.get("opportunitiesData") or data.get("data") or []
            break
        except Exception as e:
            last_err = e
            continue
    if records is None and last_err:
        raise last_err
    return records or []
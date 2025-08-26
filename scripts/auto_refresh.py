# scripts/auto_refresh.py
import os, sys
from pathlib import Path
from datetime import datetime, timezone

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import sqlalchemy as sa
from sqlalchemy import create_engine, text as sqltext

import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")

# Handle SAM_KEYS as comma OR newline separated
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]
print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")

# Time window (default last 24h)
try:
    DAYS_BACK = int(os.environ.get("DAYS_BACK", "1"))
except ValueError:
    DAYS_BACK = 1
print(f"DAYS_BACK = {DAYS_BACK}")

# Paging controls (tune via GitHub Actions env if desired)
PAGE_SIZE = int(os.environ.get("SAM_PAGE_SIZE", "200"))      # how many per page
MAX_RECORDS = int(os.environ.get("SAM_MAX_RECORDS", "5000")) # safety cap per run
print(f"Paging: PAGE_SIZE={PAGE_SIZE}, MAX_RECORDS={MAX_RECORDS}")

# --- DB (add connect timeout for Postgres) ---
pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}  # seconds

engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)
print("auto_refresh.py: engine created", flush=True)

# Quick DB ping so we can see if we hang here
try:
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))
    print("auto_refresh.py: DB ping OK", flush=True)
except Exception as e:
    print("auto_refresh.py: DB ping FAILED:", repr(e), flush=True)
    sys.exit(2)

# ---------- Insert helper ----------
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

def insert_new_records_only(records) -> int:
    if not records:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in records:
        # Fast path: don't fetch description here; backfill runs separately
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=False)
        if (m.get("notice_type") or "").strip().lower() == "justification":
            continue
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        row = {k: (m.get(k) or "") for k in COLS_TO_SAVE}
        row["pulled_at"] = now_iso
        rows.append(row)

    if not rows:
        print("No new rows to insert after filtering/mapping.")
        return 0

    cols = ["pulled_at"] + COLS_TO_SAVE
    placeholders = ", ".join(":" + c for c in cols)
    sql = sa.text(f"""
        INSERT INTO solicitationraw ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (notice_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def db_counts():
    try:
        with engine.connect() as conn:
            total = conn.execute(sqltext("SELECT COUNT(*) FROM solicitationraw")).scalar_one()
            max_pulled = conn.execute(sqltext("SELECT MAX(pulled_at) FROM solicitationraw")).scalar_one()
        return int(total or 0), str(max_pulled or "")
    except Exception as e:
        print("db_counts() failed:", repr(e))
        return None, None

# ---------- Main ----------
def main():
    print("auto_refresh.py: entered main()", flush=True)
    print("Starting auto-refresh job...")
    total_before, last_pulled = db_counts()
    if total_before is not None:
        print(f"DB before: {total_before} rows; last pulled_at: {last_pulled}")

    try:
        print("Fetching solicitations from SAM.gov with pagination…", flush=True)

        total_fetched = 0
        total_inserted = 0
        offset = 0
        page_idx = 0

        while total_fetched < MAX_RECORDS:
            page_idx += 1
            print(f"  → Page {page_idx}: offset={offset}, limit={PAGE_SIZE}", flush=True)

            raw = gs.get_sam_raw_v3(
                days_back=DAYS_BACK,
                limit=PAGE_SIZE,
                api_keys=SAM_KEYS,
                filters={},
                offset=offset,   # requires get_sam_raw_v3 to accept 'offset'
            )

            count = len(raw)
            total_fetched += count
            print(f"    fetched {count} records (cumulative fetched: {total_fetched})", flush=True)

            if count == 0:
                break

            # Log a few IDs from the first pages to verify paging works
            if page_idx <= 2:
                try:
                    sample_ids = [
                        gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=False).get("notice_id")
                        for r in raw[:5]
                    ]
                    print(f"    sample notice_ids: {sample_ids}")
                except Exception as e:
                    print("    sample logging failed:", repr(e))

            inserted = insert_new_records_only(raw)
            total_inserted += inserted
            print(f"    inserted {inserted} new (cumulative inserted: {total_inserted})", flush=True)

            if count < PAGE_SIZE:
                # last page
                break

            offset += PAGE_SIZE

        print(f"Done. Total fetched: {total_fetched}, total inserted: {total_inserted}", flush=True)

        total_after, last_pulled2 = db_counts()
        if total_after is not None:
            print(f"DB after: {total_after} rows; last pulled_at: {last_pulled2}")

        if total_inserted == 0 and total_fetched == 0:
            print("DEBUG: SAM.gov returned 0 items. Possible reasons: no new postings in window, "
                  "too-narrow server-side filters, or upstream hiccup.")

        print("Auto-refresh job completed.")

    except SamQuotaError:
        print("ERROR: SAM.gov quota exceeded.")
        sys.exit(2)
    except SamAuthError:
        print("ERROR: SAM.gov auth failed. Check SAM_KEYS secret.")
        sys.exit(2)
    except SamBadRequestError as e:
        print(f"ERROR: Bad request to SAM.gov: {e}")
        sys.exit(2)
    except Exception as e:
        print("Auto refresh failed:", repr(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
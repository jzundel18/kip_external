# scripts/backfill_descriptions.py
import os, sys, time
from pathlib import Path
from datetime import datetime

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import pandas as pd
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

# --- DB (with timeout for Postgres) ---
pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}

engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)

# ---------- Helpers ----------
def fetch_missing(limit: int = 50) -> pd.DataFrame:
    """
    Return up to `limit` rows with missing/empty descriptions.
    Use sa.text(...) so named params like :lim compile correctly for psycopg2.
    Order by pulled_at (always present) so it works on both SQLite and Postgres.
    """
    stmt = sa.text("""
        SELECT notice_id, title, link
        FROM solicitationraw
        WHERE description IS NULL OR description = ''
        ORDER BY pulled_at DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(stmt, conn, params={"lim": int(limit)})
    return df

def update_description(notice_id: str, desc: str):
    stmt = sa.text("UPDATE solicitationraw SET description = :desc WHERE notice_id = :nid")
    with engine.begin() as conn:
        conn.execute(stmt, {"desc": desc, "nid": notice_id})

# ---------- Main ----------
def main():
    print("backfill_descriptions.py: starting…", flush=True)

    # Quick DB ping so logs tell us if we’re stuck on DB
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        print("DB ping OK")
    except Exception as e:
        print("DB ping FAILED:", repr(e))
        sys.exit(2)

    # Pull a small batch of missing descriptions
    try:
        df = fetch_missing(limit=50)
    except Exception as e:
        print("DB fetch failed:", repr(e))
        sys.exit(2)

    if df.empty:
        print("No rows missing descriptions.")
        return

    print(f"Found {len(df)} solicitations missing descriptions.")

    updated = 0
    for i, row in df.iterrows():
        nid = str(row["notice_id"])
        title = str(row.get("title") or "")
        link = str(row.get("link") or "")

        try:
            # Fetch fresh detail from SAM.gov by notice_id.
            recs = gs.get_sam_raw_v3(
                days_back=30,   # wide window to find the item
                limit=1,
                api_keys=SAM_KEYS,
                filters={"notice_id": nid}
            )
            if not recs:
                print(f"[{i}] {nid} — no record returned from SAM.gov")
                continue

            rec = recs[0]
            mapped = gs.map_record_allowed_fields(rec, api_keys=SAM_KEYS, fetch_desc=True)
            desc = (mapped.get("description") or "").strip()

            if desc:
                update_description(nid, desc)
                updated += 1
                print(f"[{i}] {nid} — description updated ({len(desc)} chars).")
            else:
                print(f"[{i}] {nid} — still no description.")
        except (SamQuotaError, SamAuthError, SamBadRequestError) as e:
            print(f"[{i}] {nid} — SAM error:", repr(e))
            # If quota error, don’t hammer the API further
            if isinstance(e, SamQuotaError):
                break
        except Exception as e:
            print(f"[{i}] {nid} — error fetching description:", repr(e))

        # polite delay
        time.sleep(0.5)

    print(f"Backfill complete. {updated} descriptions updated.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Backfill script crashed:", repr(e))
        sys.exit(1)
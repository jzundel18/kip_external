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
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sqltext

import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")

# SAM_KEYS: accept comma/newline separated
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r","").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]
print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")

# Backfill knobs (env overrideable)
BATCH_SIZE = int(os.environ.get("BACKFILL_BATCH", "50"))      # rows per DB batch
MAX_UPDATES = int(os.environ.get("BACKFILL_MAX", "500"))      # safety cap per run
DELAY_SEC = float(os.environ.get("BACKFILL_DELAY_SEC", "0.5"))# polite delay between API calls

# --- DB (with timeout for Postgres) ---
pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}
engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)

def fetch_missing(limit: int) -> pd.DataFrame:
    """
    Return up to `limit` rows with missing/empty descriptions.
    NOTE: use sqltext() for named bind parameter portability.
    """
    sql = sqltext("""
        SELECT notice_id, title, link
        FROM solicitationraw
        WHERE (description IS NULL OR description = '')
        ORDER BY posted_date DESC NULLS LAST, pulled_at DESC NULLS LAST
        LIMIT :lim
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"lim": limit})

def update_description(notice_id: str, desc: str):
    sql = sa.text("UPDATE solicitationraw SET description = :desc WHERE notice_id = :nid")
    with engine.begin() as conn:
        conn.execute(sql, {"desc": desc, "nid": notice_id})

def main():
    print("backfill_descriptions.py: starting…", flush=True)
    total_updated = 0
    batches = 0

    while total_updated < MAX_UPDATES:
        try:
            df = fetch_missing(limit=BATCH_SIZE)
        except Exception as e:
            print("DB fetch failed:", repr(e))
            sys.exit(2)

        if df.empty:
            print(f"No rows missing descriptions. Total updated this run: {total_updated}")
            break

        print(f"Batch {batches+1}: found {len(df)} rows missing descriptions…")

        for i, row in df.iterrows():
            if total_updated >= MAX_UPDATES:
                print(f"Reached MAX_UPDATES={MAX_UPDATES}; stopping.")
                break

            nid = str(row["notice_id"])
            title = str(row.get("title") or "")
            link = str(row.get("link") or "")

            try:
                # Try to fetch detail — prefer v2 detail, fallback to v1 noticedesc.
                # This returns "" if no description could be fetched (not 'None').
                desc = gs.fetch_notice_description(nid, SAM_KEYS).strip()

                if desc:
                    update_description(nid, desc)
                    total_updated += 1
                    print(f"  [{total_updated}] {nid} — description updated ({len(desc)} chars).")
                else:
                    # Leave it empty; another run may succeed later
                    print(f"  [skip] {nid} — no description returned yet.")

            except SamQuotaError:
                print("  [quota] All SAM.gov keys rate-limited. Stopping backfill early.")
                print(f"Backfill partial complete. {total_updated} descriptions updated.")
                sys.exit(0)
            except SamAuthError:
                print("  [auth] SAM.gov auth failed for all keys. Stopping.")
                sys.exit(2)
            except SamBadRequestError as e:
                print(f"  [400] Bad request for {nid}: {e}")
            except Exception as e:
                print(f"  [error] {nid} — {repr(e)}")

            time.sleep(DELAY_SEC)

        batches += 1

    print(f"Backfill complete. {total_updated} descriptions updated in {batches} batch(es).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Backfill script crashed:", repr(e))
        sys.exit(1)
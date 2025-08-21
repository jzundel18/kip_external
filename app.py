import os
import re
import json
from typing import Optional, List, Dict, Any
from sqlalchemy import inspect
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
import streamlit as st
from datetime import date
from sqlmodel import SQLModel, Field, Session, create_engine, select

import numpy as np
from openai import OpenAI
def ai_make_blurbs(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_items: int = 200,
) -> dict[str, str]:
    """
    Returns {notice_id: blurb}. Each blurb is a super short, plain-English summary
    of what the solicitation is for (title + description distilled).
    """
    if df is None or df.empty:
        return {}

    # Build compact payload (cap items to keep prompt small)
    cols = ["notice_id", "title", "description"]
    use = df[[c for c in cols if c in df.columns]].head(max_items).copy()
    items = []
    for _, r in use.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": (r.get("title") or "")[:300],
            "description": (r.get("description") or "")[:2000],
        })

    system_msg = (
        "You are helping a contracts analyst. For each item, write a single, "
        "very short blurb (max ~12 words) summarizing what the solicitation buys/needs. "
        "Plain English, no fluff, no NAICS/set-aside boilerplate, no agency names, "
        "no punctuation at the end if not needed."
    )
    user_msg = {
        "items": items,
        "format": 'Return JSON: {"blurbs":[{"notice_id":"...","blurb":"..."}]} with the same order.'
    }

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        out = {}
        for row in data.get("blurbs", []):
            nid = str(row.get("notice_id", "")).strip()
            blurb = (row.get("blurb") or "").strip()
            if nid and blurb:
                out[nid] = blurb
        return out
    except Exception as e:
        st.warning(f"Could not generate blurbs right now ({e}). Showing titles instead.")
        return {}
def ai_downselect_df(company_desc: str, df: pd.DataFrame, api_key: str,
                     threshold: float = 0.20, top_k: int | None = None) -> pd.DataFrame:
    """
    Compute cosine similarity between company_desc and each row's (title + description).
    Keep rows with similarity >= threshold. Optionally keep only top_k.
    """
    if df.empty:
        return df

    # Build texts to embed
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).str.slice(0, 3000).tolist()

    try:
        client = OpenAI(api_key=api_key)
        # Get embeddings for company & rows (text-embedding-3-small is cheap & good)
        q = client.embeddings.create(model="text-embedding-3-small", input=[company_desc])
        Xq = np.array(q.data[0].embedding, dtype=np.float32)

        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        X = np.array([d.embedding for d in r.data], dtype=np.float32)

        # cosine similarity
        Xq_norm = Xq / (np.linalg.norm(Xq) + 1e-9)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sims = X_norm @ Xq_norm

        df = df.copy()
        df["ai_score"] = sims

        if top_k is not None and top_k > 0:
            df = df.sort_values("ai_score", ascending=False).head(int(top_k))
        else:
            df = df[df["ai_score"] >= float(threshold)].sort_values("ai_score", ascending=False)

        return df.reset_index(drop=True)

    except Exception as e:
        st.warning(f"AI downselect unavailable right now ({e}). Falling back to simple keyword filter.")
        # crude fallback: any word from company desc appearing in title/description
        kws = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{4,}", company_desc)]
        if not kws:
            return df
        blob = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
        mask = blob.apply(lambda t: any(k in t for k in kws))
        return df[mask].reset_index(drop=True)
# =========================
# Streamlit page & helpers
# =========================
st.set_page_config(page_title="GovContract Assistant MVP", layout="wide")

def normalize_naics_input(text_in: str) -> list[str]:
    if not text_in:
        return []
    values = re.split(r"[,\s]+", text_in.strip())
    return [v for v in (re.sub(r"[^\d]", "", x) for x in values) if v]

def parse_keywords_or(text_in: str) -> list[str]:
    return [k.strip() for k in text_in.split(",") if k.strip()]

# =========================
# Password gate
# =========================
APP_PW = st.secrets.get("APP_PASSWORD", "")
def gate():
    if not APP_PW:
        return True
    if st.session_state.get("auth_ok"):
        return True
    pw = st.text_input("Enter access password", type="password")
    if pw and pw.strip() == APP_PW.strip():
        st.session_state.auth_ok = True
        st.rerun()
    st.stop()
if not gate():
    st.stop()

# =========================
# Secrets
# =========================
def get_secret(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
SERP_API_KEY   = get_secret("SERP_API_KEY")
SAM_KEYS       = get_secret("SAM_KEYS", [])

if isinstance(SAM_KEYS, str):
    SAM_KEYS = [k.strip() for k in SAM_KEYS.split(",") if k.strip()]
elif not isinstance(SAM_KEYS, (list, tuple)):
    SAM_KEYS = []

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SERP_API_KEY": SERP_API_KEY,
    "SAM_KEYS": SAM_KEYS,
}.items() if not v]
if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

# =========================
# Database (Supabase or SQLite)
# =========================
DB_URL = st.secrets.get("SUPABASE_DB_URL") or "sqlite:///app.db"

if DB_URL.startswith("postgresql+psycopg2://"):
    engine = create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        connect_args={
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
else:
    engine = create_engine(DB_URL, pool_pre_ping=True)

# Connectivity check (clean—no host/user printed)
try:
    with engine.connect() as conn:
        ver = conn.execute(sa.text("select version()")).first()
    st.sidebar.success("✅ Connected to database")
    if ver and isinstance(ver, tuple):
        st.sidebar.caption(ver[0])
except Exception as e:
    st.sidebar.error("❌ Database connection failed")
    st.sidebar.exception(e)
    st.stop()

# =========================
# Static schema: ONLY the attributes you want to persist
# =========================
class SolicitationRaw(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    pulled_at: Optional[str] = Field(default=None, index=True)
    notice_id: str = Field(index=True, nullable=False, unique=True)

    solicitation_number: Optional[str] = None
    title: Optional[str] = None
    notice_type: Optional[str] = None

    posted_date: Optional[str] = Field(default=None, index=True)
    response_date: Optional[str] = Field(default=None, index=True)
    archive_date: Optional[str] = Field(default=None, index=True)
    naics_code: Optional[str] = Field(default=None, index=True)

    set_aside_code: Optional[str] = Field(default=None, index=True)

    description: Optional[str] = None
    link: Optional[str] = None

# Create table
# --- create base table defined by SQLModel ---
SQLModel.metadata.create_all(engine)

# --- lightweight migration: ensure required columns & unique index exist ---
REQUIRED_COLS = {
    "pulled_at": "TEXT",
    "notice_id": "TEXT",
    "solicitation_number": "TEXT",
    "title": "TEXT",
    "notice_type": "TEXT",
    "posted_date": "TEXT",
    "response_date": "TEXT",
    "archive_date": "TEXT",
    "naics_code": "TEXT",
    "set_aside_code": "TEXT",
    "description": "TEXT",
    "link": "TEXT",
}

try:
    insp = inspect(engine)
    existing_cols = {c["name"] for c in insp.get_columns("solicitationraw")}
    missing_cols = [c for c in REQUIRED_COLS if c not in existing_cols]

    if missing_cols:
        with engine.begin() as conn:
            for col in missing_cols:
                # explicit quoting to handle lowercase names
                conn.execute(sa.text(f'ALTER TABLE solicitationraw ADD COLUMN "{col}" {REQUIRED_COLS[col]}'))

    # Create unique index on notice_id if not present (Postgres supports IF NOT EXISTS)
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_solicitationraw_notice_id
            ON solicitationraw (notice_id)
        """))

except Exception as e:
    # If you're on SQLite or hit a permission quirk, show a helpful error
    st.warning(f"Migration note: {e}")

# =========================
# Import your modules
# =========================
import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError
import find_relevant_suppliers as fs
import generate_proposal as gp

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.markdown("---")
    st.markdown("### Feed Settings")
    max_results_refresh = st.number_input(
        "Max results when refreshing feed",
        min_value=1, max_value=2000, value=500, step=50,
        help="How many solicitations to pull from SAM.gov when you click Refresh."
    )
    st.markdown("---")
    st.subheader("Tips")
    st.write("• Refresh writes only brand-new notice_ids to the DB (no updates).")
    st.write("• Filter below uses DB only (no extra SAM calls).")

# =========================
# Upsert-only-new helpers
# =========================
# exactly what you store now
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]
def insert_new_records_only(records) -> int:
    if not records:
        return 0

    rows = []
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)        
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        rows.append({k: (m.get(k) or "") for k in COLS_TO_SAVE})

    if not rows:
        return 0

    sql = sa.text(f"""
        INSERT INTO solicitationraw (
            {", ".join(COLS_TO_SAVE)}
        ) VALUES (
            {", ".join(":"+c for c in COLS_TO_SAVE)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)  # bulk insert

    return len(rows)
DISPLAY_COLS = [
    "pulled_at","notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code","description","link"
]

def query_filtered_df(filters: dict) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql_query(f"SELECT {', '.join(DISPLAY_COLS)} FROM solicitationraw", conn)

    if df.empty:
        return df

    # keyword OR filter (guard for missing description/title just in case)
    kws = [k.lower() for k in (filters.get("keywords_or") or []) if k]
    if kws:
        title = df["title"].fillna("")
        desc  = df["description"].fillna("")
        blob = (title + " " + desc).str.lower()
        df = df[blob.apply(lambda t: any(k in t for k in kws))]

    # NAICS filter
    naics = [re.sub(r"[^\d]","", x) for x in (filters.get("naics") or []) if x]
    if naics:
        df = df[df["naics_code"].isin(naics)]

    # set-aside filter
    sas = filters.get("set_asides") or []
    if sas:
        df = df[df["set_aside_code"].fillna("").str.lower().apply(lambda s: any(sa.lower() in s for sa in sas))]

    # notice types
    nts = filters.get("notice_types") or []
    if nts:
        df = df[df["notice_type"].fillna("").str.lower().apply(lambda s: any(nt.lower() in s for nt in nts))]

    # due before
    due_before = filters.get("due_before")
    if due_before:
        dd = pd.to_datetime(df["response_date"], errors="coerce", utc=True)
        df = df[dd.dt.date <= pd.to_datetime(due_before).date()]

    return df.reset_index(drop=True)

# =========================
# Header & top controls
# =========================
st.title("GovContract Assistant MVP")
st.caption("Only storing required SAM fields; inserts brand-new notices only (no updates).")

colR1, colR2 = st.columns([1,1])
with colR1:
    if st.button("🔄 Refresh today's feed"):
        try:
            raw = gs.get_sam_raw_v3(
                days_back=0,
                limit=int(max_results_refresh),
                api_keys=SAM_KEYS,
                filters={}
            )
            n = insert_new_records_only(raw)
            st.success(f"Attempted inserts: {n} (existing notice_ids are skipped).")
        except SamQuotaError:
            st.warning("SAM.gov quota likely exceeded on all provided keys. Try again after daily reset or add more keys.")
        except SamBadRequestError as e:
            st.error(f"Bad request to SAM.gov (check date/params): {e}")
        except SamAuthError:
            st.error("All SAM.gov keys failed (auth/network). Double-check your keys in Secrets.")
        except Exception as e:
            st.exception(e)

with colR2:
    # Show count of rows currently in DB
    try:
        with engine.connect() as conn:
            cnt = pd.read_sql_query("SELECT COUNT(*) AS c FROM solicitationraw", conn)["c"].iloc[0]
        st.metric("Rows in DB", int(cnt))
    except Exception:
        st.metric("Rows in DB", 0)

# =========================
# Session state
# =========================
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None

import json
from openai import OpenAI

def ai_rank_solicitations_by_fit(
    df: pd.DataFrame,
    company_desc: str,
    api_key: str,
    top_k: int = 5,
    max_candidates: int = 100,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Returns: [{"notice_id": "...", "score": 0-100, "reason": "..."}] ordered best→worst.
    """
    if df is None or df.empty:
        return []

    cols_we_care = [
        "notice_id", "title", "description", "naics_code",
        "set_aside_code", "response_date", "posted_date", "link"
    ]
    df2 = df[[c for c in cols_we_care if c in df.columns]].copy().head(max_candidates)

    items = []
    for _, r in df2.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": str(r.get("title", ""))[:300],
            "description": str(r.get("description", ""))[:3000],
            "naics_code": str(r.get("naics_code", "")),
            "set_aside_code": str(r.get("set_aside_code", "")),
            "response_date": str(r.get("response_date", "")),
            "posted_date": str(r.get("posted_date", "")),
            "link": str(r.get("link", "")),
        })

    system_msg = (
        "You are a contracts analyst. Rank solicitations by how well they match the company description. "
        "Consider title, description, NAICS, set-aside, and due date recency. Prefer clear technical/mission fit."
    )
    user_msg = {
        "company_description": company_desc,
        "solicitations": items,
        "instructions": (
            f"Return the top {top_k} as JSON: "
            '{"ranked":[{"notice_id":"...","score":0-100,"reason":"..."}]}. '
            "Score reflects strength of fit (higher is better). Keep reasons short and specific."
        ),
    }

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    # Parse JSON safely
    try:
        data = json.loads(content or "{}")
        ranked = data.get("ranked", [])
    except Exception:
        return []

    # Keep only rows that exist in df2; sort and dedupe
    keep_ids = set(df2["notice_id"].astype(str).tolist())
    cleaned = []
    for item in ranked:
        nid = str(item.get("notice_id", ""))
        if nid in keep_ids:
            cleaned.append({
                "notice_id": nid,
                "score": float(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            })

    seen, out = set(), []
    for x in cleaned:
        if x["notice_id"] not in seen:
            seen.add(x["notice_id"])
            out.append(x)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["1) Fetch Solicitations", "2) Supplier Suggestions", "3) Proposal Draft"])

# ---- Tab 1
with tab1:
    st.header("Filter Solicitations")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        limit_results = st.number_input("Max results to show", min_value=1, max_value=5000, value=200)
    with colB:
        keywords_raw = st.text_input("Filter keywords (OR, comma-separated)", value="rfq, rfp, rfi")
    with colC:
        naics_raw = st.text_input("Filter by NAICS (comma-separated)", value="")

    with st.expander("More filters (optional)"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            set_asides = st.multiselect("Set-aside code", ["SBA","WOSB","EDWOSB","HUBZone","SDVOSB","8A","SDB"])
        with col2:
            due_before = st.date_input("Due before (optional)", value=None, format="YYYY-MM-DD")
        with col3:
            notice_types = st.multiselect(
                "Notice types",
                ["Solicitation","Combined Synopsis/Solicitation","Sources Sought","Special Notice","SRCSGT","RFI"]
            )

    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
    }
    st.subheader("Company profile (optional)")
    company_desc = st.text_area("Brief company description (for AI downselect)", value="", height=120)
    use_ai_downselect = st.checkbox("Use AI to downselect based on description", value=False)

    # One-button flow: manual filters always; optional AI ranking (top 5) when checkbox + description
    if st.button("Show top results", type="primary", key="btn_show_results"):
        try:
            # 1) Apply manual filters from DB (no SAM calls)
            df = query_filtered_df(filters)

            if df.empty:
                st.warning("No solicitations match your filters. Try adjusting filters or refresh today's feed.")
            else:
                # --- NEW: ask AI for tiny blurbs to show instead of raw titles
                blurbs = ai_make_blurbs(df, OPENAI_API_KEY, model="gpt-4o-mini", max_items=200)
                if blurbs:
                    df = df.copy()
                    df["blurb"] = df["notice_id"].astype(str).map(blurbs).fillna(df["title"].fillna(""))
                    # Put blurb up front for visibility in the table
                    front = ["blurb"]
                    rest = [c for c in df.columns if c not in front]
                    df = df[front + rest]
                st.session_state.sol_df = df
                st.success(f"Found {len(df)} solicitations.")

                # 2) If AI downselect is checked & description present, rank & show top 5 (title-first, expandable details)
                if use_ai_downselect and company_desc.strip():
                    with st.spinner("Ranking top matches with AI…"):
                        ranked = ai_rank_solicitations_by_fit(
                            df=st.session_state.sol_df,
                            company_desc=company_desc.strip(),
                            api_key=OPENAI_API_KEY,
                            top_k=5,
                            max_candidates=100,   # adjust cost/speed as you like
                            model="gpt-4o-mini",
                        )

                    if not ranked:
                        st.info("AI ranking returned no results; showing the manually filtered table instead.")
                        # Show the table
                        show_df = st.session_state.sol_df.head(int(limit_results)) if limit_results else st.session_state.sol_df
                        st.subheader(f"Solicitations ({len(show_df)})")
                        st.dataframe(show_df, use_container_width=True)
                        st.download_button(
                            "Download filtered as CSV",
                            show_df.to_csv(index=False).encode("utf-8"),
                            file_name="sol_list.csv",
                            mime="text/csv"
                        )
                    else:
                        # Build a quick lookup for details by notice_id
                        base_df = st.session_state.sol_df
                        idx_by_id = {str(x): i for i, x in enumerate(base_df["notice_id"].astype(str).tolist())}

                        # Compose a small DataFrame for download (top-5 ranked rows)
                        top_rows = []
                        st.success(f"Top {len(ranked)} matches by company fit:")
                        for i, item in enumerate(ranked, start=1):
                            nid = item["notice_id"]
                            score = int(round(item.get("score", 0)))
                            reason = item.get("reason", "")
                            row = base_df.iloc[idx_by_id[nid]]

                            # accumulate for download
                            top_rows.append(row)

                            hdr = (row.get("blurb") or row.get("title") or "Untitled")
                            with st.expander(f"{i}. {hdr}  —  Score {score}/100"):
                                st.write(f"**Notice Type:** {row.get('notice_type','')}")
                                st.write(f"**Posted:** {row.get('posted_date','')}")
                                st.write(f"**Response Due:** {row.get('response_date','')}")
                                st.write(f"**NAICS:** {row.get('naics_code','')}")
                                st.write(f"**Set-aside:** {row.get('set_aside_code','')}")
                                link = row.get("link","")
                                if link:
                                    st.write(f"**Link:** {link}")
                                if reason:
                                    st.markdown("**Why this matched (AI):**")
                                    st.info(reason)

                        # Offer download of just the top-5
                        if top_rows:
                            top_df = pd.DataFrame(top_rows).reset_index(drop=True)
                            st.download_button(
                                "Download Top-5 (AI-ranked) as CSV",
                                top_df.to_csv(index=False).encode("utf-8"),
                                file_name="top5_ai_ranked.csv",
                                mime="text/csv"
                            )

                else:
                    # 3) No AI requested; show the manually filtered table (capped by limit_results)
                    show_df = st.session_state.sol_df.head(int(limit_results)) if limit_results else st.session_state.sol_df
                    st.subheader(f"Solicitations ({len(show_df)})")
                    st.dataframe(show_df, use_container_width=True)
                    st.download_button(
                        "Download filtered as CSV",
                        show_df.to_csv(index=False).encode("utf-8"),
                        file_name="sol_list.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.exception(e)
# ---- Tab 2
with tab2:
    st.header("Find Supplier Suggestions")
    st.write("This uses your solicitation rows + Google results (via SerpAPI) to propose suppliers and rough quotes.")
    our_rec = st.text_input("Favored suppliers (comma-separated)", value="")
    our_not = st.text_input("Do-not-use suppliers (comma-separated)", value="")
    max_google = st.number_input("Max Google results per item", min_value=1, max_value=20, value=5)

    if st.button("Run supplier suggestion", type="primary"):
        if st.session_state.sol_df is None:
            st.error("Load or fetch solicitations in Tab 1 first.")
        else:
            sol_dicts = st.session_state.sol_df.to_dict(orient="records")
            favored = [x.strip() for x in our_rec.split(",") if x.strip()]
            not_favored = [x.strip() for x in our_not.split(",") if x.strip()]
            try:
                results = fs.get_suppliers(
                    solicitations=sol_dicts,
                    our_recommended_suppliers=favored,
                    our_not_recommended_suppliers=not_favored,
                    Max_Google_Results=int(max_google),
                    OpenAi_API_Key=OPENAI_API_KEY,
                    Serp_API_Key=SERP_API_KEY
                )
                sup_df = pd.DataFrame(results)
                st.session_state.sup_df = sup_df
                st.success(f"Generated {len(sup_df)} supplier rows.")
            except Exception as e:
                st.exception(e)

    if st.session_state.sup_df is not None:
        st.subheader("Supplier suggestions")
        to_show = st.session_state.sol_df.copy()
        for col in ["notice_id", "description"]:
            if col in to_show.columns:
                to_show = to_show.drop(columns=[col])

        st.dataframe(to_show, use_container_width=True)
        st.download_button(
            "Download as CSV",
            st.session_state.sup_df.to_csv(index=False).encode("utf-8"),
            file_name="supplier_suggestions.csv",
            mime="text/csv"
        )

# ---- Tab 3
with tab3:
    st.header("Generate Proposal Draft")
    st.write("Select one or more supplier-suggestion rows and generate a proposal draft using your templates.")
    bid_template = st.text_input("Bid template file path (DOCX or TXT)", value="/mnt/data/BID_TEMPLATE.docx")
    solinfo_template = st.text_input("Solicitation info template (DOCX or TXT)", value="/mnt/data/SOLICITATION_INFO_TEMPLATE.docx")
    out_dir = st.text_input("Output directory", value="/mnt/data/proposals")

    uploaded_sup2 = st.file_uploader("Or upload supplier_suggestions.csv here", type=["csv"], key="sup_upload2")
    if uploaded_sup2 is not None:
        try:
            df_upload = pd.read_csv(uploaded_sup2)
            st.session_state.sup_df = df_upload
            st.success(f"Loaded {len(df_upload)} supplier suggestions from upload.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.sup_df is not None:
        st.dataframe(st.session_state.sup_df, use_container_width=True)
        idxs = st.multiselect(
            "Pick rows to draft",
            options=list(range(len(st.session_state.sup_df))),
            help="Leave empty to draft all"
        )
        if st.button("Generate proposal(s)", type="primary"):
            os.makedirs(out_dir, exist_ok=True)
            try:
                df_sel = st.session_state.sup_df.iloc[idxs] if idxs else st.session_state.sup_df
                gp.validate_supplier_and_write_proposal(
                    df=df_sel,
                    output_directory=out_dir,
                    Open_AI_API_Key=OPENAI_API_KEY,
                    BID_TEMPLATE_FILE=bid_template,
                    SOl_INFO_TEMPLATE=solinfo_template
                )
                st.success(f"Drafted proposals to {out_dir}.")
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.caption("DB schema is fixed to only the required SAM fields. Refresh inserts brand-new notices only (no updates).")


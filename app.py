# app.py

import os
import re
import json
from typing import Optional, List, Dict, Any
from sqlalchemy import inspect
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
import streamlit as st
from datetime import date, datetime, timezone
from sqlmodel import SQLModel, Field, create_engine
import numpy as np
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import find_relevant_suppliers as fs
import generate_proposal as gp

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="GovContract Assistant MVP", layout="wide")

# =========================
# Small helpers
# =========================
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

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
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
# Static schema (only the fields you want)
# =========================
class SolicitationRaw(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    pulled_at: Optional[str] = Field(default=None, index=True)  # when we pulled from SAM
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
SQLModel.metadata.create_all(engine)

# Lightweight migration: ensure columns & unique index
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
                conn.execute(sa.text(f'ALTER TABLE solicitationraw ADD COLUMN "{col}" {REQUIRED_COLS[col]}'))

    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_solicitationraw_notice_id
            ON solicitationraw (notice_id)
        """))
except Exception as e:
    st.warning(f"Migration note: {e}")

# ---------------------------
# Company directory (new table)
# ---------------------------
class Company(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = Field(default=None, index=True)

# Create (or update) tables
SQLModel.metadata.create_all(engine)

# Lightweight migration for Companies (safe if already exists)
try:
    insp = inspect(engine)
    if "company" in [t.lower() for t in insp.get_table_names()]:
        existing_cols = {c["name"] for c in insp.get_columns("company")}
        REQUIRED_COMPANY_COLS = {
            "name": "TEXT",
            "description": "TEXT",
            "city": "TEXT",
            "state": "TEXT",
        }
        missing_cols = [c for c in REQUIRED_COMPANY_COLS if c not in existing_cols]
        if missing_cols:
            with engine.begin() as conn:
                for col in missing_cols:
                    conn.execute(sa.text(f'ALTER TABLE company ADD COLUMN "{col}" {REQUIRED_COMPANY_COLS[col]}'))
except Exception as e:
    st.warning(f"Company table migration note: {e}")

with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.caption("Feed refresh runs automatically (no manual refresh needed).")
    st.markdown("---")

# =========================
# AI helpers
# =========================

def _embed_texts(texts: list[str], api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2 normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X

@st.cache_data(show_spinner=False, ttl=3600)
def cached_company_embeddings(_companies: pd.DataFrame, api_key: str) -> dict:
    """
    Returns {"df": companies_df, "X": normalized embeddings (np.ndarray)}.
    Cache invalidates if companies data changes (we use df contents as key).
    """
    if _companies.empty:
        return {"df": _companies, "X": np.zeros((0, 1536), dtype=np.float32)}
    texts = _companies["description"].fillna("").astype(str).tolist()
    X = _embed_texts(texts, api_key)
    return {"df": _companies.copy(), "X": X}

def ai_identify_gaps(company_desc: str, solicitation_text: str, api_key: str) -> str:
    """
    Ask the model to identify key capability gaps we'd need to fill to bid solo.
    Returns a short paragraph (1–3 sentences).
    """
    client = OpenAI(api_key=api_key)
    sys = "You are a federal contracting expert. Be concise and specific."
    user = (
        "Company description:\n"
        f"{company_desc}\n\n"
        "Solicitation (title+description):\n"
        f"{solicitation_text[:6000]}\n\n"
        "List the biggest capability gaps this company would need to fill to bid competitively. "
        "Return a short paragraph (no bullets)."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"Gap identification unavailable ({e}).")
        return ""

def pick_best_partner_for_gaps(gap_text: str, companies: pd.DataFrame, api_key: str, top_n: int = 1) -> pd.DataFrame:
    """
    Use embeddings to match companies to the gap text. Returns top_n rows from `companies`.
    """
    if companies.empty or not gap_text.strip():
        return companies.head(0)
    # Embed gap_text
    q = _embed_texts([gap_text], api_key)[0]  # normalized
    # Embed companies (cached)
    emb = cached_company_embeddings(companies, api_key)
    dfc, X = emb["df"], emb["X"]
    if X.shape[0] == 0:
        return dfc.head(0)
    sims = X @ q  # cosine similarity
    dfc = dfc.copy()
    dfc["score"] = sims
    return dfc.sort_values("score", ascending=False).head(top_n)

def ai_partner_justification(company_row: dict, solicitation_text: str, gap_text: str, api_key: str) -> dict:
    """
    Returns {"justification": "...", "joint_proposal": "..."} short blurbs.
    """
    client = OpenAI(api_key=api_key)
    sys = (
        "You are a federal contracts strategist. Be concise, concrete, and persuasive. "
        "You MUST reply with a single JSON object only."
    )
    # IMPORTANT: include the word "JSON" and the exact shape
    instructions = (
        'Return ONLY a JSON object of the form: '
        '{"justification":"one short sentence", "joint_proposal":"one short sentence"} '
        '— no markdown, no extra text.'
    )
    user_payload = {
        "partner_company": {
            "name": company_row.get("name",""),
            "capabilities": company_row.get("description",""),
            "location": f'{company_row.get("city","")}, {company_row.get("state","")}'.strip(", "),
        },
        "our_capability_gaps": gap_text,
        "solicitation": solicitation_text[:6000],
        "instructions": instructions,
    }

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                # include "JSON" in the user message content as well
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.2,
        )
        content = (r.choices[0].message.content or "").strip()
        data = json.loads(content or "{}")
        j = str(data.get("justification","")).strip()
        jp = str(data.get("joint_proposal","")).strip()
        if not j and not jp:
            # graceful fallback
            return {"justification": "No justification returned.", "joint_proposal": ""}
        return {"justification": j, "joint_proposal": jp}
    except Exception as e:
        return {"justification": f"Justification unavailable ({e})", "joint_proposal": ""}
    
def ai_downselect_df(company_desc: str, df: pd.DataFrame, api_key: str,
                     threshold: float = 0.20, top_k: int | None = None) -> pd.DataFrame:
    """
    Embedding-based similarity between company_desc and (title + description).
    Keep rows with similarity >= threshold, or top_k if provided.
    """
    if df.empty:
        return df

    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).str.slice(0, 2000).tolist()
    try:
        client = OpenAI(api_key=api_key)
        q = client.embeddings.create(model="text-embedding-3-small", input=[company_desc])
        Xq = np.array(q.data[0].embedding, dtype=np.float32)

        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        X = np.array([d.embedding for d in r.data], dtype=np.float32)

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
        kws = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{4,}", company_desc)]
        if not kws:
            return df
        blob = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
        mask = blob.apply(lambda t: any(k in t for k in kws))
        return df[mask].reset_index(drop=True)

def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def ai_make_blurbs(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_items: int = 160,
    chunk_size: int = 40,
) -> dict[str, str]:
    """
    Returns {notice_id: blurb}. Short plain-English summaries of solicitations.
    Batches requests to keep prompts small & reliable.
    """
    if df is None or df.empty:
        return {}

    cols = ["notice_id", "title", "description"]
    use = df[[c for c in cols if c in df.columns]].head(max_items).copy()

    # Prepare items (truncate to keep prompt tight)
    items = []
    for _, r in use.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": (r.get("title") or "")[:200],
            "description": (r.get("description") or "")[:800],  # keep short to avoid token bloat
        })

    client = OpenAI(api_key=api_key)
    out: dict[str, str] = {}

    system_msg = (
        "You are helping a contracts analyst. For each item, write one very short, "
        "plain-English blurb (~8–12 words) summarizing what the solicitation buys/needs. "
        "Avoid agency names, set-aside boilerplate, and extra punctuation."
    )

    for batch in _chunk(items, chunk_size):
        user_msg = {
            "items": batch,
            "format": 'Return JSON: {"blurbs":[{"notice_id":"...","blurb":"..."}]} in the same order.'
        }
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
            for row in data.get("blurbs", []):
                nid = str(row.get("notice_id", "")).strip()
                blurb = (row.get("blurb") or "").strip()
                if nid and blurb:
                    out[nid] = blurb
        except Exception as e:
            # If a batch fails, skip it but continue with others
            st.warning(f"Could not generate blurbs for one batch ({e}).")
            continue

    return out

# =========================
# DB helpers
# =========================
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

DISPLAY_COLS = [
    # For UI we will *exclude* notice_id and description, but they remain selectable below
    "pulled_at",
    "solicitation_number",
    "notice_type",
    "posted_date",
    "response_date",
    "naics_code",
    "set_aside_code",
    "link",
    # (notice_id and description are intentionally not shown in UI, but exist in the DF).
]

def insert_new_records_only(records) -> int:
    """
    Maps raw SAM records, adds pulled_at, and inserts only new notice_ids.
    """
    if not records:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)
        if (m.get("notice_type") or "").strip().lower() == "justification":
            continue
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        row = {k: (m.get(k) or "") for k in COLS_TO_SAVE}
        row["pulled_at"] = now_iso
        rows.append(row)

    if not rows:
        return 0

    sql = sa.text(f"""
        INSERT INTO solicitationraw (
            pulled_at, {", ".join(COLS_TO_SAVE)}
        ) VALUES (
            :pulled_at, {", ".join(":"+c for c in COLS_TO_SAVE)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)  # bulk insert

    return len(rows)

def query_filtered_df(filters: dict) -> pd.DataFrame:
    # Pull a superset of columns; we'll hide some in the UI
    base_cols = ["pulled_at","notice_id","solicitation_number","title","notice_type",
                 "posted_date","response_date","archive_date",
                 "naics_code","set_aside_code","description","link"]

    with engine.connect() as conn:
        df = pd.read_sql_query(f"SELECT {', '.join(base_cols)} FROM solicitationraw", conn)

    if df.empty:
        return df

    # keyword OR filter
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

def companies_df() -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(
                "SELECT id, name, description, city, state FROM company ORDER BY name",
                conn
            )
        except Exception:
            return pd.DataFrame(columns=["id","name","description","city","state"])

def insert_company_row(row: dict) -> None:
    sql = sa.text("""
        INSERT INTO company (name, description, city, state)
        VALUES (:name, :description, :city, :state)
    """)
    row = {k: (row.get(k) or "") for k in ["name","description","city","state"]}
    row["created_at"] = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(sql, row)

def bulk_insert_companies(df: pd.DataFrame) -> int:
    needed = ["name","description","city","state"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    rows = df[needed].fillna("").to_dict(orient="records")
    for r in rows:
        insert_company_row(r)
    return len(rows)



def _hide_notice_and_description(df: pd.DataFrame) -> pd.DataFrame:
    # UI should not show these two columns
    return df.drop(columns=[c for c in ["notice_id", "description"] if c in df.columns], errors="ignore")
# =========================
# Header & top controls
# =========================
st.title("GovContract Assistant MVP")
st.caption("Only storing required SAM fields; inserts brand-new notices only (no updates).")

colR1, colR2 = st.columns([2,1])
with colR1:
    st.info("Feed updates automatically every hour.")
with colR2:
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

# =========================
# AI ranker (used for the Top 5 expander section)
# =========================
def ai_rank_solicitations_by_fit(
    df: pd.DataFrame,
    company_desc: str,
    api_key: str,
    top_k: int = 5,
    max_candidates: int = 100,
    model: str = "gpt-4o-mini",
) -> list[dict]:
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
            "description": str(r.get("description", ""))[:1500],
            "naics_code": str(r.get("naics_code", "")),
            "set_aside_code": str(r.get("set_aside_code", "")),
            "response_date": str(r.get("response_date", "")),
            "posted_date": str(r.get("posted_date", "")),
            "link": str(r.get("link", "")),
        })

    system_msg = (
        "You are a contracts analyst. Rank solicitations by how well they match the company description. "
        "Consider title, description, NAICS, set-aside, and due date recency."
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

    try:
        data = json.loads(content or "{}")
        ranked = data.get("ranked", [])
    except Exception:
        return []

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
tab1, tab2, tab3, tab4 = st.tabs([
    "1) Fetch Solicitations",
    "2) Supplier Suggestions",
    "3) Proposal Draft",
    "4) Partner Matches"
])
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
    st.session_state.company_desc = company_desc
    use_ai_downselect = st.checkbox("Use AI to downselect based on description", value=False)
    
    if st.button("Show top results", type="primary", key="btn_show_results"):
        try:
            # 1) Apply manual filters from DB (no SAM calls)
            df = query_filtered_df(filters)

            if df.empty:
                st.warning("No solicitations match your filters. Try adjusting filters or refresh today's feed.")
                st.session_state.sol_df = None
            else:
                # ===== IF AI downselect + company description → Rank Top 5 =====
                if use_ai_downselect and company_desc.strip():
                    # Pre-trim with embeddings to keep prompt small & fast
                    # Keep the most-similar N items before LLM ranking
                    pretrim = ai_downselect_df(company_desc.strip(), df, OPENAI_API_KEY, top_k=80)

                    if pretrim.empty:
                        st.info("AI pre-filter returned nothing. Showing manually filtered table instead.")
                        show_df = df.head(int(limit_results)) if limit_results else df
                        # Generate very short blurbs only for rows we will display
                        blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                                max_items=min(150, int(limit_results or 150)))
                        show_df = show_df.copy()
                        show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))
                        st.session_state.sol_df = show_df
                        st.subheader(f"Solicitations ({len(show_df)})")
                        st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
                        st.download_button(
                            "Download filtered as CSV",
                            show_df.to_csv(index=False).encode("utf-8"),
                            file_name="sol_list.csv",
                            mime="text/csv"
                        )
                    else:
                        # Rank Top 5 with LLM (small prompt, capped candidates)
                        with st.spinner("Ranking top matches with AI…"):
                            ranked = ai_rank_solicitations_by_fit(
                                df=pretrim,
                                company_desc=company_desc.strip(),
                                api_key=OPENAI_API_KEY,
                                top_k=5,
                                max_candidates=60,     # keep prompt compact → faster, cheaper
                                model="gpt-4o-mini",
                            )

                        if not ranked:
                            st.info("AI ranking returned no results; showing the manually filtered table instead.")
                            show_df = df.head(int(limit_results)) if limit_results else df
                            # blurbs only for what we show
                            blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                                    max_items=min(150, int(limit_results or 150)))
                            show_df = show_df.copy()
                            show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))
                            st.session_state.sol_df = show_df
                            st.subheader(f"Solicitations ({len(show_df)})")
                            st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
                            st.download_button(
                                "Download filtered as CSV",
                                show_df.to_csv(index=False).encode("utf-8"),
                                file_name="sol_list.csv",
                                mime="text/csv"
                            )
                        else:
                            # Build ordered Top-5 dataframe
                            id_order = [x["notice_id"] for x in ranked]
                            preorder = {nid: i for i, nid in enumerate(id_order)}
                            top_df = pretrim[pretrim["notice_id"].astype(str).isin(id_order)].copy()
                            top_df["__order"] = top_df["notice_id"].astype(str).map(preorder)
                            top_df = top_df.sort_values("__order").drop(columns="__order")

                            # Generate blurbs only for Top-5
                            blurbs = ai_make_blurbs(top_df, OPENAI_API_KEY, model="gpt-4o-mini", max_items=10)
                            top_df["blurb"] = top_df["notice_id"].astype(str).map(blurbs).fillna(top_df["title"].fillna(""))

                            # Show Top-5 as expanders: blurb first, click to see details
                            st.success(f"Top {len(top_df)} matches by company fit:")
                            # quick lookup for reasons and scores
                            reason_by_id = {x["notice_id"]: x.get("reason", "") for x in ranked}
                            score_by_id  = {x["notice_id"]: x.get("score", 0)  for x in ranked}

                            # add a fit_score column so Tab 4 can auto-filter moderate/strong matches
                            top_df["fit_score"] = top_df["notice_id"].astype(str).map(score_by_id).fillna(0).astype(float)

                            for i, row in enumerate(top_df.itertuples(index=False), start=1):
                                hdr = (getattr(row, "blurb", None) or getattr(row, "title", None) or "Untitled")
                                with st.expander(f"{i}. {hdr}"):
                                    st.write(f"**Notice Type:** {getattr(row, 'notice_type', '')}")
                                    st.write(f"**Posted:** {getattr(row, 'posted_date', '')}")
                                    st.write(f"**Response Due:** {getattr(row, 'response_date', '')}")
                                    st.write(f"**NAICS:** {getattr(row, 'naics_code', '')}")
                                    st.write(f"**Set-aside:** {getattr(row, 'set_aside_code', '')}")
                                    link = getattr(row, "link", "")
                                    if link:
                                        st.write(f"**Link:** {link}")
                                    reason = reason_by_id.get(str(getattr(row, "notice_id", "")), "")
                                    if reason:
                                        st.markdown("**Why this matched (AI):**")
                                        st.info(reason)

                            # Save the Top-5 dataframe itself for Tab 4
                            st.session_state.top5_df = top_df.reset_index(drop=True)
                            # Remember for downstream tabs / downloads
                            st.session_state.sol_df = top_df.copy()
                            # Invalidate any old partner matches (Tab 4 will auto-rebuild)
                            st.session_state.partner_matches = None
                            st.session_state.top5_stamp = datetime.utcnow().isoformat()
                            # Optional: let user download just the Top-5 rows
                            st.download_button(
                                "Download Top-5 (AI-ranked) as CSV",
                                top_df.to_csv(index=False).encode("utf-8"),
                                file_name="top5_ai_ranked.csv",
                                mime="text/csv"
                            )

                # ===== NO AI → just show the filtered table with blurbs =====
                else:
                    show_df = df.head(int(limit_results)) if limit_results else df
                    # blurbs only for what we show (kept small)
                    blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                            max_items=min(150, int(limit_results or 150)))
                    show_df = show_df.copy()
                    show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))

                    st.session_state.sol_df = show_df
                    st.subheader(f"Solicitations ({len(show_df)})")
                    st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
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
        st.dataframe(st.session_state.sup_df, use_container_width=True)
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
# ---- Tab 4
with tab4:
    st.header("Partner Matches (from Top-5)")

    # Load companies from DB (read-only; no manage UI)
    df_companies = companies_df()

    # Need Top-5 from Tab 1
    top5 = st.session_state.get("top5_df")

    if top5 is None or top5.empty:
        st.info("No Top-5 available. In Tab 1, run AI ranking to generate Top-5 first.")
    elif df_companies.empty:
        st.info("Your company database is empty. Populate the 'company' table in Supabase with: name, description, city, state.")
    else:
        # Reuse company description from Tab 1 (stored there)
        company_desc_global = (st.session_state.get("company_desc") or "").strip()
        if not company_desc_global:
            st.info("No company description provided in Tab 1. Please enter one there and rerun.")
        else:
            # Auto-compute matches when Top-5 changes or cache is empty
            need_recompute = (
                st.session_state.get("partner_matches") is None or
                st.session_state.get("partner_matches_stamp") != st.session_state.get("top5_stamp")
            )

            if need_recompute:
                with st.spinner("Analyzing gaps and selecting partners for Top-5…"):
                    matches = []
                    for _, row in top5.head(5).iterrows():
                        title = str(row.get("title", "")) or "Untitled"
                        blurb = str(row.get("blurb", "")).strip()
                        desc  = str(row.get("description", "")) or ""
                        sol_text = f"{title}\n\n{desc}"

                        # 1) Identify our capability gaps for this solicitation
                        gaps = ai_identify_gaps(company_desc_global, sol_text, OPENAI_API_KEY)

                        # 2) Pick best partner from company DB to fill those gaps
                        best = pick_best_partner_for_gaps(gaps or sol_text, df_companies, OPENAI_API_KEY, top_n=1)
                        if best.empty:
                            matches.append({
                                "title": title,
                                "blurb": blurb,
                                "partner": None,
                                "gaps": gaps,
                                "ai": {"justification": "No suitable partner found.", "joint_proposal": ""}
                            })
                            continue

                        partner = best.iloc[0].to_dict()

                        # 3) Short justification + joint-proposal sketch (JSON-safe)
                        ai = ai_partner_justification(partner, sol_text, gaps, OPENAI_API_KEY)

                        matches.append({
                            "title": title,
                            "blurb": blurb,
                            "partner": partner,
                            "gaps": gaps,
                            "ai": ai
                        })

                # Cache results with a stamp tied to the Top-5
                st.session_state.partner_matches = matches
                st.session_state.partner_matches_stamp = st.session_state.get("top5_stamp")

            # Render cached matches
            matches = st.session_state.get("partner_matches", [])
            if not matches:
                st.info("No partner matches computed yet.")
            else:
                for m in matches:
                    hdr = (m.get("blurb") or m.get("title") or "Untitled").strip()
                    partner_name = (m.get("partner") or {}).get("name", "")
                    exp_title = f"Opportunity: {hdr}"
                    if partner_name:
                        exp_title += f" — Partner: {partner_name}"

                    with st.expander(exp_title):
                        # Partner block
                        if m.get("partner"):
                            p = m["partner"]
                            loc = ", ".join([x for x in [p.get("city",""), p.get("state","")] if x])
                            st.markdown("**Recommended Partner:**")
                            st.write(f"{p.get('name','')}" + (f" — {loc}" if loc else ""))
                        else:
                            st.warning("No suitable partner found for this opportunity.")

                        # Gaps
                        if m.get("gaps"):
                            st.markdown("**Why we need a partner (our capability gaps):**")
                            st.write(m["gaps"])

                        # Why this partner
                        just = (m.get("ai", {}) or {}).get("justification", "")
                        if just:
                            st.markdown("**Why this partner:**")
                            st.info(just)

                        # Joint proposal idea
                        jp = (m.get("ai", {}) or {}).get("joint_proposal", "").strip()
                        if jp:
                            st.markdown("**Targeted joint proposal idea:**")
                            st.write(jp)
st.markdown("---")
st.caption("DB schema is fixed to only the required SAM fields. Refresh inserts brand-new notices only (no updates).")
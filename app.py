import os
import re
import json
from typing import Optional, List, Dict, Any

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from datetime import date
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy.engine.url import make_url

import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# =========================
# Streamlit page & helpers
# =========================
st.set_page_config(page_title="GovContract Assistant MVP", layout="wide")

def normalize_naics_input(text: str) -> list[str]:
    if not text:
        return []
    values = re.split(r"[,\s]+", text.strip())
    return [v for v in (re.sub(r"[^\d]", "", x) for x in values) if v]

def parse_keywords_or(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]

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

# Normalize SAM_KEYS so it is always a list
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

# Single authoritative connectivity check
try:
    parsed = make_url(DB_URL)
    st.sidebar.write("DB host:", parsed.host)
    st.sidebar.write("DB user:", parsed.username)
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
# Model
# =========================
class SolicitationRaw(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    notice_id: str = Field(index=True)
    notice_type: Optional[str] = None
    solicitation_number: Optional[str] = None
    title: Optional[str] = None
    posted_date: Optional[str] = Field(default=None, index=True)
    due_date: Optional[str] = Field(default=None, index=True)
    naics_code: Optional[str] = Field(default=None, index=True)
    set_aside: Optional[str] = Field(default=None, index=True)
    agency: Optional[str] = Field(default=None, index=True)
    link: Optional[str] = None
    description: Optional[str] = None

    # Keep full SAM record as JSON text (works on SQLite & Postgres)
    raw_json: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))

# Create table once connected
try:
    SQLModel.metadata.create_all(engine)
except Exception as e:
    st.sidebar.error("DB init error while creating tables")
    st.sidebar.exception(e)
    st.stop()

# =========================
# Import your modules
# =========================
import get_relevant_solicitations as gs
import find_relevant_suppliers as fs
import generate_proposal as gp

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.success("✅ API keys loaded from Secrets")

    st.markdown("### Feed Settings")
    max_results_refresh = st.number_input(
        "Max results when refreshing feed",
        min_value=50, max_value=2000, value=500, step=50,
        help="How many solicitations to pull from SAM.gov when you click Refresh."
    )

    st.markdown("---")
    st.subheader("Tips")
    st.write("• Start with moderate limits (100–500) while testing.")
    st.write("• Use DB filters below; refresh only when needed.")

# =========================
# DB helpers
# =========================
def _table_to_df(rows: List[SolicitationRaw]) -> pd.DataFrame:
    return pd.DataFrame([{
        "notice id": r.notice_id,
        "notice type": r.notice_type,
        "solicitation number": r.solicitation_number,
        "title": r.title,
        "posted date": r.posted_date,
        "due date": r.due_date,
        "NAICS Code": r.naics_code,
        "set-aside": r.set_aside,
        "agency": r.agency,
        "solicitation link": r.link,
        "item description": r.description,
    } for r in rows])

def upsert_raw_records(records: List[Dict[str, Any]]) -> int:
    """
    Map key fields for filtering/display and store full JSON as raw_json.
    Upsert by notice_id.
    """
    if not records:
        return 0

    def g(obj: dict, *keys, default=""):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default

    inserted = 0
    with Session(engine) as s:
        for r in records:
            notice_id = str(g(r, "noticeId", "id", default="")).strip()
            if not notice_id:
                continue

            exists = s.exec(select(SolicitationRaw).where(SolicitationRaw.notice_id == notice_id)).first()

            mapped = dict(
                notice_id=notice_id,
                notice_type=str(g(r, "noticeType", "type", default="") or ""),
                solicitation_number=str(g(r, "solicitationNumber", "solnum", default="") or ""),
                title=str(g(r, "title", default="") or ""),
                posted_date=str(g(r, "postedDate", "publishDate", default="") or ""),
                due_date=str(g(r, "responseDate", "closeDate", default="") or ""),
                naics_code=str(g(r, "naics", "naicsCode", default="") or ""),
                set_aside=str(g(r, "setAside", "setAsideCode", default="") or ""),
                agency=str(g(r, "department", "agency", "organizationName", default="") or ""),
                link=str(g(r, "url", "samLink", default="") or ""),
                description=str(g(r, "description", "synopsis", default="") or ""),
                raw_json=json.dumps(r),
            )

            if not exists:
                s.add(SolicitationRaw(**mapped))
                inserted += 1
            else:
                for k, v in mapped.items():
                    setattr(exists, k, v)
        s.commit()
    return inserted

def refresh_todays_feed(limit: int = 500) -> int:
    """
    Pull today's records from SAM.gov (limit=N) and upsert all fields (raw_json + mapped columns).
    """
    filters = {
        "keywords_or": [],
        "naics": [],
        "set_asides": [],
        "agency_contains": "",
        "due_before": None,
        "notice_types": [],
    }
    raw_records = gs.get_sam_raw_v3(days_back=0, limit=int(limit), api_keys=SAM_KEYS, filters=filters)
    return upsert_raw_records(raw_records)

def query_today_from_db(filters: dict) -> pd.DataFrame:
    """
    Query the DB (no SAM calls) and apply UI filters client-side on the mapped columns.
    """
    kws = [k.lower() for k in (filters.get("keywords_or") or []) if k]
    naics = [re.sub(r"[^\d]", "", x) for x in (filters.get("naics") or []) if x]
    sas = filters.get("set_asides") or []
    agency_contains = (filters.get("agency_contains") or "").lower().strip()
    due_before = filters.get("due_before")
    notice_types = filters.get("notice_types") or []

    with Session(engine) as s:
        stmt = select(SolicitationRaw).where(sa.true())

        if naics:
            stmt = stmt.where(SolicitationRaw.naics_code.in_(naics))

        if sas:
            stmt = stmt.where(SolicitationRaw.set_aside.in_(sas))

        if notice_types:
            ors = [SolicitationRaw.notice_type.ilike(f"%{nt}%") for nt in notice_types]
            stmt = stmt.where(sa.or_(*ors))

        if agency_contains:
            stmt = stmt.where(SolicitationRaw.agency.ilike(f"%{agency_contains}%"))

        rows = s.exec(stmt).all()

    df = _table_to_df(rows)
    if df.empty:
        return df

    # Keywords OR (title + description)
    if kws:
        blob = (df["title"].fillna("") + " " + df["item description"].fillna("")).str.lower()
        df = df[blob.apply(lambda t: any(k in t for k in kws))]

    # Due before (parse as date)
    if due_before and "due date" in df.columns:
        dd = pd.to_datetime(df["due date"], errors="coerce", utc=True)
        df = df[dd.dt.date <= pd.to_datetime(due_before).date()]

    return df.reset_index(drop=True)

# =========================
# Header & top controls
# =========================
st.title("GovContract Assistant MVP")
st.caption("Bid matching, suppliers, and proposal drafting — backed by a persistent database.")

st.info("Refresh pulls from SAM.gov once → everything else filters the local database (no extra API calls).")

colR1, colR2, colR3 = st.columns([1,1,1])
with colR1:
    if st.button("🔄 Refresh today's feed"):
        try:
            n = refresh_todays_feed(limit=max_results_refresh)
            st.success(f"Refreshed from SAM.gov. Upserted ~{n} rows (limit {max_results_refresh}).")
        except SamQuotaError as e:
            st.warning("SAM.gov quota likely exceeded on all provided keys. Try again after daily reset or add more keys.")
        except SamBadRequestError as e:
            st.error(f"Bad request to SAM.gov: {e}")
        except SamAuthError as e:
            st.error("All SAM.gov keys failed (auth/network). Double-check your keys in Secrets.")
        except Exception as e:
            st.exception(e)

with colR2:
    # Show count of rows currently in DB (cross-version safe)
    with Session(engine) as s:
        res = s.exec(select(sa.func.count(SolicitationRaw.id)))
        val = res.first()
        total = int(val[0]) if isinstance(val, tuple) else int(val or 0)
    st.metric("Rows in DB", f"{total}")

with colR3:
    if st.button("⬇️ Download entire DB as CSV"):
        with Session(engine) as s:
            rows = s.exec(select(SolicitationRaw)).all()
        df_all = _table_to_df(rows)
        if df_all.empty:
            st.warning("Database is empty.")
        else:
            st.download_button(
                "Download solicitations.csv",
                df_all.to_csv(index=False).encode("utf-8"),
                file_name="solicitations.csv",
                mime="text/csv"
            )

# =========================
# Session state
# =========================
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["1) Fetch Solicitations", "2) Supplier Suggestions", "3) Proposal Draft"])

# ---- Tab 1
with tab1:
    st.header("Fetch Relevant Solicitations")

    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        days_back = st.number_input("Days back", min_value=0, max_value=120, value=0,
            help="Currently used only when fetching from SAM.gov; 0 = today.")
    with colB:
        limit_results = st.number_input("Max results (UI filter only)", min_value=1, max_value=2000, value=200)
    with colC:
        keywords_raw = st.text_input("Filter keywords (OR, comma-separated)", value="rfq, rfp, rfi")
    with colD:
        naics_raw = st.text_input("Filter by NAICS (comma-separated)", value="")

    with st.expander("More filters (optional)"):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            set_asides = st.multiselect("Set-aside", ["SB","WOSB","EDWOSB","HUBZone","SDVOSB","8A","SDB"])
        with col2:
            agency_contains = st.text_input("Agency contains", value="")
        with col3:
            due_before = st.date_input("Due before (optional)", value=None, format="YYYY-MM-DD")
        with col4:
            notice_types = st.multiselect(
                "Notice types",
                ["Solicitation","Combined Synopsis/Solicitation","Sources Sought","Special Notice","SRCSGT","RFI"]
            )

    st.subheader("Company profile (optional)")
    company_desc = st.text_area("Brief company description (for AI downselect – disabled in this build)", value="", height=120)
    use_ai_downselect = st.checkbox("Use AI to downselect based on description (coming soon)", value=False, disabled=True)

    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "agency_contains": agency_contains.strip(),
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
    }

    if st.button("Fetch from local DB (today's feed)", type="primary"):
        try:
            df = query_today_from_db(filters)
            if limit_results and not df.empty:
                df = df.head(int(limit_results))
            if df.empty:
                st.warning("No solicitations match your filters. Try adjusting filters or refresh today's feed.")
            else:
                st.session_state.sol_df = df
                st.success(f"Found {len(df)} solicitations.")
        except Exception as e:
            st.exception(e)

    if st.session_state.sol_df is not None:
        st.subheader("Solicitations")
        st.dataframe(st.session_state.sol_df, use_container_width=True)
        st.download_button(
            "Download as CSV",
            st.session_state.sol_df.to_csv(index=False).encode("utf-8"),
            file_name="sol_list.csv",
            mime="text/csv"
        )

        # Optional: inspect raw JSON of a selected notice
        ids = st.session_state.sol_df["notice id"].tolist()
        pick = st.selectbox("Inspect raw JSON for notice id:", options=["(select)"] + ids)
        if pick != "(select)":
            with Session(engine) as s:
                rec = s.exec(select(SolicitationRaw).where(SolicitationRaw.notice_id == pick)).first()
            if rec and rec.raw_json:
                try:
                    st.json(json.loads(rec.raw_json))
                except Exception:
                    st.code(rec.raw_json)

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
                files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
                if files:
                    st.write("Generated files:")
                    for f in files:
                        st.write(os.path.join(out_dir, f))
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.caption("This MVP stores full SAM.gov records (raw_json) and exposes key filters on indexed columns.")
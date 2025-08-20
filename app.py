import os
import re
from typing import Optional

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from datetime import date, datetime, timezone

from sqlmodel import SQLModel, Field, Session, create_engine, select

# ---------------------------
# Page & small helpers
# ---------------------------
st.set_page_config(page_title="GovContract Assistant MVP", layout="wide")

def normalize_naics_input(text: str) -> list[str]:
    if not text:
        return []
    values = re.split(r"[,\s]+", text.strip())
    return [v for v in (re.sub(r"[^\d]", "", x) for x in values) if v]

def parse_keywords_or(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]

# ---------------------------
# Password gate
# ---------------------------
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

# ---------------------------
# Centralized secrets
# ---------------------------
def get_secret(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERP_API_KEY   = get_secret("SERP_API_KEY")
SAM_KEYS       = get_secret("SAM_KEYS", [])

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SERP_API_KEY": SERP_API_KEY,
    "SAM_KEYS": SAM_KEYS,
}.items() if not v]
if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

# ---------------------------
# DB setup (SQLite by default; Supabase if provided)
# ---------------------------
DB_URL = st.secrets.get("SUPABASE_DB_URL") or "sqlite:///app.db"
engine = create_engine(DB_URL, pool_pre_ping=True)

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

try:
    SQLModel.metadata.create_all(engine)
except Exception as e:
    st.error(f"DB init error: {e}")

# ---------------------------
# Imports for your modules (repo-local)
# ---------------------------
import get_relevant_solicitations as gs
import find_relevant_suppliers as fs
import generate_proposal as gp

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.markdown("---")
    st.subheader("Tips")
    st.write("• Start with small limits (10–50) while testing.")

# === Add this new section anywhere after the above ===
with st.sidebar:
    st.markdown("### Feed Settings")
    max_results_refresh = st.number_input(
        "Max results when refreshing feed",
        min_value=50, max_value=2000, value=500, step=50,
        help="This sets the limit on how many solicitations to pull from SAM.gov when refreshing today's feed."
    )

# === Your existing Dev Mode section stays as is ===
with st.sidebar:
    st.markdown("### Dev Mode")
    DEV_MODE = st.checkbox("Use sample CSV instead of SAM.gov", value=True,
                           help="When ON, the refresh button will load from sample_feed.csv and not call SAM.gov.")
    SAMPLE_CSV_PATH = st.text_input("Sample CSV path", value="sample_feed.csv",
                                    help="File used in Dev Mode for refreshing today's feed.")

with st.sidebar.expander("🔍 Debug Database"):
    if st.button("Show first 20 rows from DB"):
        with Session(engine) as s:
            rows = s.exec(select(SolicitationRaw).limit(20)).all()
        dbg_df = pd.DataFrame([r.__dict__ for r in rows])
        st.dataframe(dbg_df, use_container_width=True)

# ---------------------------
# Small DB helpers
# ---------------------------
def _table_to_df(rows):
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

def export_db_to_sample_csv(path="sample_feed.csv"):
    with Session(engine) as s:
        rows = s.exec(select(SolicitationRaw)).all()
    df = _table_to_df(rows)
    df.to_csv(path, index=False)
    return len(df)

def load_df_into_db(df: pd.DataFrame) -> int:
    inserted = 0
    required_cols = {
        "notice id","notice type","solicitation number","title","posted date","due date",
        "NAICS Code","set-aside","agency","solicitation link","item description"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Sample CSV missing required columns: {sorted(missing_cols)}")

    with Session(engine) as s:
        for _, r in df.iterrows():
            notice_id = str(r["notice id"]).strip()
            if not notice_id:
                continue
            exists = s.exec(
                select(SolicitationRaw).where(SolicitationRaw.notice_id == notice_id)
            ).first()

            if not exists:
                s.add(SolicitationRaw(
                    notice_id=notice_id,
                    notice_type=str(r.get("notice type","") or ""),
                    solicitation_number=str(r.get("solicitation number","") or ""),
                    title=str(r.get("title","") or ""),
                    posted_date=str(r.get("posted date","") or ""),
                    due_date=str(r.get("due date","") or ""),
                    naics_code=str(r.get("NAICS Code","") or ""),
                    set_aside=str(r.get("set-aside","") or ""),
                    agency=str(r.get("agency","") or ""),
                    link=str(r.get("solicitation link","") or ""),
                    description=str(r.get("item description","") or ""),
                ))
                inserted += 1
            else:
                exists.notice_type = str(r.get("notice type","") or "")
                exists.solicitation_number = str(r.get("solicitation number","") or "")
                exists.title = str(r.get("title","") or "")
                exists.posted_date = str(r.get("posted date","") or "")
                exists.due_date = str(r.get("due date","") or "")
                exists.naics_code = str(r.get("NAICS Code","") or "")
                exists.set_aside = str(r.get("set-aside","") or "")
                exists.agency = str(r.get("agency","") or "")
                exists.link = str(r.get("solicitation link","") or "")
                exists.description = str(r.get("item description","") or "")
        s.commit()
    return inserted

# ---------------------------
# Refresh today's feed (Dev Mode aware)
# ---------------------------
def refresh_todays_feed(limit: int = 500, cap_dev_csv: bool = True) -> int:
    """
    Refresh today's feed into the local DB.

    When DEV_MODE is True:
      - Loads SAMPLE_CSV_PATH instead of calling SAM.gov.
      - If cap_dev_csv is True, only the first `limit` rows of the CSV are loaded (handy for testing).

    When DEV_MODE is False:
      - Calls SAM.gov once using get_relevant_solicitations_v2 with days_back=0 and the given `limit`.
      - Maps the response into the same schema our DB expects and upserts.

    Returns:
      int: number of rows inserted/updated (best-effort; based on upserts).
    """
    # ---------- DEV MODE: use local CSV, no API calls ----------
    if DEV_MODE:
        if not os.path.exists(SAMPLE_CSV_PATH):
            raise FileNotFoundError(f"Sample CSV not found: {SAMPLE_CSV_PATH}")
        df = pd.read_csv(SAMPLE_CSV_PATH)
        if cap_dev_csv and isinstance(limit, int) and limit > 0:
            df = df.head(limit)
        return load_df_into_db(df)

    # ---------- REAL MODE: call SAM.gov once ----------
    filters = {
        "keywords_or": [],
        "naics": [],
        "set_asides": [],
        "agency_contains": "",
        "due_before": None,
        "notice_types": [],
        "use_ai_downselect": False,
        "company_desc": "",
    }

    list_final = gs.get_relevant_solicitations_v2(
        days_back=0,                 # "today" only
        limit=int(limit),            # pulled from sidebar "Feed Settings"
        api_keys=SAM_KEYS,           # from secrets
        filters=filters,
        openai_api_key=OPENAI_API_KEY
    )

    if not list_final:
        return 0

    header, rows = list_final[0], list_final[1:]
    # Build a header index for safe lookups
    h = {c: i for i, c in enumerate(header)}

    # Convert to the DataFrame schema our DB loader expects
    data = []
    for r in rows:
        data.append({
            "notice id":            r[h.get("notice id", 0)] if len(r) > h.get("notice id", 0) else "",
            "notice type":          r[h.get("notice type", 0)] if len(r) > h.get("notice type", 0) else "",
            "solicitation number":  r[h.get("solicitation number", 0)] if len(r) > h.get("solicitation number", 0) else "",
            "title":                r[h.get("title", 0)] if len(r) > h.get("title", 0) else "",
            "posted date":          r[h.get("posted date", 0)] if len(r) > h.get("posted date", 0) else "",
            "due date":             r[h.get("due date", 0)] if len(r) > h.get("due date", 0) else "",
            "NAICS Code":           r[h.get("NAICS Code", 0)] if len(r) > h.get("NAICS Code", 0) else "",
            "set-aside":            r[h.get("set-aside", 0)] if len(r) > h.get("set-aside", 0) else "",
            "agency":               r[h.get("agency", 0)] if len(r) > h.get("agency", 0) else "",
            "solicitation link":    r[h.get("solicitation link", 0)] if len(r) > h.get("solicitation link", 0) else "",
            "item description":     r[h.get("item description", 0)] if len(r) > h.get("item description", 0) else "",
        })

    df = pd.DataFrame(data)
    return load_df_into_db(df)

def query_today_from_db(filters: dict) -> pd.DataFrame:
    """Query the local DB and apply UI filters (no SAM calls)."""
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

# ---------------------------
# Header & global controls
# ---------------------------
st.title("GovContract Assistant MVP")
st.caption("A simple UI around your existing scripts for bid matching, suppliers, and proposal drafting.")

st.info("This app queries today's feed from the local database.")
# Sidebar control for feed limit

colR1, colR2 = st.columns([1,1])
with colR1:
    if st.button("🔄 Refresh today's feed"):
        try:
            n = refresh_todays_feed(limit=max_results_refresh)
            src = "sample CSV" if DEV_MODE else "SAM.gov"
            st.success(f"Refreshed from {src}. Upserted ~{n} rows (limit {max_results_refresh}).")
        except Exception as e:
            st.exception(e)

with colR2:
    if st.button("💾 Export DB → sample_feed.csv"):
        try:
            n = export_db_to_sample_csv(SAMPLE_CSV_PATH)
            st.success(f"Exported {n} rows to {SAMPLE_CSV_PATH}. You can now enable Dev Mode to use it.")
        except Exception as e:
            st.exception(e)

# ---------------------------
# Session state
# ---------------------------
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["1) Fetch Solicitations", "2) Supplier Suggestions", "3) Proposal Draft"])

# ---- Tab 1
with tab1:
    st.header("Fetch Relevant Solicitations")

    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        days_back = st.number_input("Days back", min_value=1, max_value=120, value=30)  # kept for future use
    with colB:
        limit_results = st.number_input("Max results", min_value=1, max_value=200, value=50)  # kept for future use
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
    company_desc = st.text_area("Brief company description (for AI downselect)", value="", height=120)
    use_ai_downselect = st.checkbox("Use AI to downselect based on description", value=False)

    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "agency_contains": agency_contains.strip(),
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
        "use_ai_downselect": bool(use_ai_downselect),
        "company_desc": company_desc.strip(),
    }

    if st.button("Fetch from local DB (today's feed)", type="primary"):
        try:
            df = query_today_from_db(filters)
            if df.empty:
                st.warning("No solicitations match your filters in today's feed. Try adjusting filters or refresh today's feed.")
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
    bid_template = st.text_input("Bid template file path (DOCX)", value="/mnt/data/BID_TEMPLATE.docx")
    solinfo_template = st.text_input("Solicitation info template (DOCX)", value="/mnt/data/SOLICITATION_INFO_TEMPLATE.docx")
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
st.caption("This is an MVP wrapper. When you're ready, we can replace CSVs with a database and add logins.")
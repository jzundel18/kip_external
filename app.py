import os
import re
import json
from typing import Optional, List, Dict, Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy import inspect
import streamlit as st
from datetime import date
from sqlmodel import SQLModel, Field, Session, create_engine, select

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

def snake(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()

def flatten_dict(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dicts/lists into a single dict with dotted keys.
    Lists are JSON-dumped to preserve content (then cast to text column).
    """
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            out.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(d, list):
        # Store lists as JSON text; we won't explode them into separate rows
        out[parent_key] = json.dumps(d)
    else:
        out[parent_key] = d
    return out

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

# Single authoritative connectivity check
try:
    parsed = make_url(DB_URL)
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
# Core model (static columns)
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
    # NOTE: No raw_json column anymore (we write every field to its own column)

# Create core table + unique index on notice_id
try:
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND indexname = 'uq_solicitationraw_notice_id'
                ) THEN
                    CREATE UNIQUE INDEX uq_solicitationraw_notice_id
                    ON solicitationraw (notice_id);
                END IF;
            EXCEPTION WHEN undefined_table THEN
                -- SQLite fallback: ignore
                NULL;
            END $$;
        """))
except Exception as e:
    # SQLite or first-time run may not support the DO block; ignore failures here.
    pass

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
# Dynamic wide-schema helpers
# =========================
TABLE_NAME = "solicitationraw"

def get_existing_columns(engine: Engine) -> set[str]:
    insp = inspect(engine)
    cols = set()
    try:
        for col in insp.get_columns(TABLE_NAME):
            cols.add(col["name"])
    except Exception:
        # Fallback: basic select *
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {TABLE_NAME} LIMIT 0"))
            cols = set(result.keys())
    return cols

def prepare_wide_columns(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten and normalize keys for each record.
    Returns list of dicts where keys are snake_case column names.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        flat = flatten_dict(r)  # dotted keys for nested
        mapped: Dict[str, Any] = {}
        for k, v in flat.items():
            col = snake(k.replace(".", "_"))
            # cast non-primitives to JSON text
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            mapped[col] = v
        # also map some canonical/duplicate fields under consistent names
        mapped.setdefault("notice_id", mapped.get("noticeid") or mapped.get("id"))
        mapped.setdefault("notice_type", mapped.get("noticetype") or mapped.get("type"))
        mapped.setdefault("solicitation_number", mapped.get("solicitationnumber") or mapped.get("solnum"))
        mapped.setdefault("title", mapped.get("title"))
        mapped.setdefault("posted_date", mapped.get("posteddate") or mapped.get("publishdate"))
        mapped.setdefault("due_date", mapped.get("responsedate") or mapped.get("closedate"))
        mapped.setdefault("naics_code", mapped.get("naics") or mapped.get("naicscode"))
        mapped.setdefault("set_aside", mapped.get("setaside") or mapped.get("setasidecode"))
        mapped.setdefault("agency", mapped.get("department") or mapped.get("agency") or mapped.get("organizationname"))
        mapped.setdefault("link", mapped.get("url") or mapped.get("samlink"))
        mapped.setdefault("description", mapped.get("description") or mapped.get("synopsis"))
        out.append(mapped)
    return out

def ensure_columns_exist(engine: Engine, sample_row: Dict[str, Any]) -> None:
    """
    Add any missing columns (TEXT) for keys in sample_row.
    We skip columns that already exist. All dynamic columns are TEXT for safety.
    """
    existing = get_existing_columns(engine)
    needed = [c for c in sample_row.keys() if c not in existing]

    if not needed:
        return

    # Build ALTER TABLE for new columns
    alters = []
    for col in needed:
        # Safety: skip illegal names
        if not re.match(r"^[a-z_][a-z0-9_]*$", col):
            continue
        alters.append(f"ADD COLUMN IF NOT EXISTS {col} TEXT")

    if not alters:
        return

    ddl = f"ALTER TABLE {TABLE_NAME} " + ", ".join(alters) + ";"
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception as e:
        # SQLite doesn't support IF NOT EXISTS in ALTER COLUMN add (older versions).
        # Fallback: try individually.
        for col in needed:
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} TEXT"))
            except Exception:
                pass  # ignore if it already exists

def upsert_wide_records(records: List[Dict[str, Any]]) -> int:
    """
    Upsert flattened records as wide columns.
    Requires a unique index on notice_id. Unknown columns are added on the fly.
    """
    if not records:
        return 0

    wide = prepare_wide_columns(records)

    # Ensure notice_id present
    wide = [w for w in wide if str(w.get("notice_id") or "").strip()]
    if not wide:
        return 0

    # Add columns as needed based on the union of keys
    # Use a representative row (union would be heavy; we'll add on the fly per row)
    # First pass: ensure core + common keys
    ensure_columns_exist(engine, wide[0])

    inserted = 0
    with engine.begin() as conn:
        for row in wide:
            ensure_columns_exist(engine, row)  # ensure any new columns for this row
            cols = list(row.keys())
            vals = [row[c] for c in cols]
            placeholders = ", ".join([f":{i}" for i in range(len(cols))])
            col_list = ", ".join(cols)

            # Build ON CONFLICT upsert (Postgres). SQLite also supports upsert with DO UPDATE.
            updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c != "notice_id"])
            sql = text(f"""
                INSERT INTO {TABLE_NAME} ({col_list})
                VALUES ({placeholders})
                ON CONFLICT (notice_id)
                DO UPDATE SET {updates}
            """)
            params = {str(i): vals[i] for i in range(len(vals))}
            conn.execute(sql, params)
            inserted += 1
    return inserted

def query_filtered_df(filters: dict) -> pd.DataFrame:
    """
    Read from DB into a DataFrame and apply light client-side filters on the
    canonical columns we maintain.
    """
    # Pull only the columns we display/filter by to keep UI responsive
    cols = [
        "notice_id","notice_type","solicitation_number","title","posted_date",
        "due_date","naics_code","set_aside","agency","link","description"
    ]
    col_list = ", ".join(cols)
    with engine.connect() as conn:
        try:
            df = pd.read_sql_query(f"SELECT {col_list} FROM {TABLE_NAME}", conn)
        except Exception:
            # Table empty or columns missing
            return pd.DataFrame(columns=cols)

    if df.empty:
        return df

    # Apply filters (OR keywords)
    kws = [k.lower() for k in (filters.get("keywords_or") or []) if k]
    if kws:
        blob = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
        df = df[blob.apply(lambda t: any(k in t for k in kws))]

    naics = [re.sub(r"[^\d]", "", x) for x in (filters.get("naics") or []) if x]
    if naics:
        df = df[df["naics_code"].isin(naics)]

    sas = filters.get("set_asides") or []
    if sas:
        df = df[df["set_aside"].fillna("").str.lower().apply(lambda s: any(sa.lower() in s for sa in sas))]

    agency_contains = (filters.get("agency_contains") or "").strip().lower()
    if agency_contains:
        df = df[df["agency"].fillna("").str.lower().str.contains(agency_contains)]

    due_before = filters.get("due_before")
    if due_before:
        dd = pd.to_datetime(df["due_date"], errors="coerce", utc=True)
        df = df[dd.dt.date <= pd.to_datetime(due_before).date()]

    return df.reset_index(drop=True)

# =========================
# Header & top controls
# =========================
st.title("GovContract Assistant MVP")
st.caption("Bid matching, suppliers, and proposal drafting — dynamic schema with one column per SAM field.")

with st.sidebar:
    st.markdown("---")
    st.subheader("Dev / Export")
    if st.button("⬇️ Download entire DB as CSV"):
        with engine.connect() as conn:
            df_all = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
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
# Top buttons
# =========================
tab_fetch, tab_suppliers, tab_proposal = st.tabs(
    ["1) Fetch Solicitations", "2) Supplier Suggestions", "3) Proposal Draft"]
)

with tab_fetch:
    st.info("Refresh pulls from SAM.gov → we upsert every field into columns. Filtering below queries the DB only.")

    colR1, colR2 = st.columns([1,1])
    with colR1:
        if st.button("🔄 Refresh today's feed"):
            try:
                raw = gs.get_sam_raw_v3(
                    days_back=0,
                    limit=int(st.session_state.get("max_results_refresh", 500) or 500),
                    api_keys=SAM_KEYS,
                    filters={}
                )
                n = upsert_wide_records(raw)
                st.success(f"Upserted ~{n} rows into DB.")
            except SamQuotaError as e:
                st.warning("SAM.gov quota likely exceeded on all provided keys. Try again after daily reset or add more keys.")
            except SamBadRequestError as e:
                st.error(f"Bad request to SAM.gov (check date/params): {e}")
            except SamAuthError as e:
                st.error("All SAM.gov keys failed (auth/network). Double-check your keys in Secrets.")
            except Exception as e:
                st.exception(e)

    with colR2:
        # Show current count of rows
        try:
            with engine.connect() as conn:
                cnt = pd.read_sql_query(f"SELECT COUNT(*) as c FROM {TABLE_NAME}", conn)["c"].iloc[0]
            st.metric("Rows in DB", int(cnt))
        except Exception:
            st.metric("Rows in DB", 0)

    # Filters
    st.header("Filter DB")
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        days_back = st.number_input("Days back (for future fetch use)", min_value=0, max_value=120, value=0, help="Not used in DB filter.")
    with colB:
        limit_results = st.number_input("Max results to show", min_value=1, max_value=5000, value=200)
    with colC:
        keywords_raw = st.text_input("Filter keywords (OR, comma-separated)", value="rfq, rfp, rfi")
    with colD:
        naics_raw = st.text_input("Filter by NAICS (comma-separated)", value="")

    with st.expander("More filters (optional)"):
        col1, col2, col3, col4 = st.columns([1,1,1,1])
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

    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "agency_contains": agency_contains.strip(),
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
    }

    if st.button("Fetch from local DB", type="primary"):
        try:
            df = query_filtered_df(filters)
            if limit_results and not df.empty:
                df = df.head(int(limit_results))
            if df.empty:
                st.warning("No solicitations match your filters. Try adjusting filters or refresh today's feed.")
            else:
                st.session_state.sol_df = df
                st.success(f"Found {len(df)} solicitations.")
        except Exception as e:
            st.exception(e)

    if st.session_state.get("sol_df") is not None:
        st.subheader("Solicitations")
        st.dataframe(st.session_state.sol_df, use_container_width=True)
        st.download_button(
            "Download filtered as CSV",
            st.session_state.sol_df.to_csv(index=False).encode("utf-8"),
            file_name="sol_list.csv",
            mime="text/csv"
        )

        # Optional: inspect a full row directly from DB (all columns)
        ids = st.session_state.sol_df["notice id"].tolist() if "notice id" in st.session_state.sol_df.columns else st.session_state.sol_df["notice_id"].tolist()
        pick = st.selectbox("Inspect full DB row (all columns) by notice_id:", options=["(select)"] + list(map(str, ids)))
        if pick != "(select)":
            with engine.connect() as conn:
                full = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE notice_id = :nid", conn, params={"nid": pick})
            if not full.empty:
                st.dataframe(full.T, use_container_width=True)

# ---- Tab 2 (Suppliers)
with tab_suppliers:
    st.header("Find Supplier Suggestions")
    st.write("This uses your solicitation rows + Google results (via SerpAPI) to propose suppliers and rough quotes.")
    our_rec = st.text_input("Favored suppliers (comma-separated)", value="")
    our_not = st.text_input("Do-not-use suppliers (comma-separated)", value="")
    max_google = st.number_input("Max Google results per item", min_value=1, max_value=20, value=5)

    if st.button("Run supplier suggestion", type="primary"):
        if st.session_state.get("sol_df") is None:
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

    if st.session_state.get("sup_df") is not None:
        st.subheader("Supplier suggestions")
        st.dataframe(st.session_state.sup_df, use_container_width=True)
        st.download_button(
            "Download as CSV",
            st.session_state.sup_df.to_csv(index=False).encode("utf-8"),
            file_name="supplier_suggestions.csv",
            mime="text/csv"
        )

# ---- Tab 3 (Proposal)
with tab_proposal:
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

    if st.session_state.get("sup_df") is not None:
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
st.caption("Dynamic schema: we add a column for every SAM.gov field we see (TEXT). Filter views use core columns for speed.")
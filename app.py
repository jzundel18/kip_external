import os
import streamlit as st

# ----- Password gate (unchanged) -----
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

# ----- Centralized secrets (keys) -----
def get_secret(name, default=None):
    # Prefer Streamlit secrets, then environment variables as fallback
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERP_API_KEY   = get_secret("SERP_API_KEY")
SAM_KEYS       = get_secret("SAM_KEYS", [])

# Validate presence once at startup
missing = [k for k,v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SERP_API_KEY": SERP_API_KEY,
    "SAM_KEYS": SAM_KEYS,
}.items() if not v]
if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

import pandas as pd
from datetime import datetime

# Import user modules
import sys
sys.path.append('/mnt/data')
import get_relevant_solicitations as gs
import find_relevant_suppliers as fs
import generate_proposal as gp

st.set_page_config(page_title="GovContract Assistant MVP", layout="wide")
st.title("GovContract Assistant MVP")
st.caption("A simple UI around your existing scripts for bid matching, suppliers, and proposal drafting.")

# --- Session state ---
if "sol_header" not in st.session_state:
    st.session_state.sol_header = None
if "sol_rows" not in st.session_state:
    st.session_state.sol_rows = None
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None

with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.markdown("---")
    st.subheader("Tips")
    st.write("• Start with small limits (10–20) while testing.")
    st.write("• You can upload existing CSVs if you prefer.")

tab1, tab2, tab3 = st.tabs(["1) Fetch Solicitations", "2) Supplier Suggestions", "3) Proposal Draft"])

# ---------------- Tab 1 ----------------
with tab1:
    st.header("Fetch Relevant Solicitations")
    colA, colB, colC = st.columns(3)
    with colA:
        days_back = st.number_input("Days back", min_value=1, max_value=120, value=30)
    with colB:
        limit_results = st.number_input("Max results", min_value=1, max_value=200, value=25)
    with colC:
        keywords_raw = st.text_input("Filter keywords (comma-separated)", value="rfq, rfp, rfi")

    uploaded_sol = st.file_uploader("Or upload an existing sol_list.csv", type=["csv"], key="sol_upload")

    def list_to_df(list_final):
        # list_final format appears to be: first element is headers, subsequent elements are rows
        if isinstance(list_final, list) and list_final and isinstance(list_final[0], list):
            header = list_final[0]
            rows = list_final[1:]
            df = pd.DataFrame(rows, columns=header)
            return df
        # Otherwise try to coerce directly
        return pd.DataFrame(list_final)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Fetch from SAM.gov", type="primary"):
            target_keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
            api_keys = SAM_KEYS  # from secrets
            try:
                list_final = gs.get_relevant_solicitation_list(
                    days_back,
                    int(limit_results),
                    api_keys,
                    target_keywords,
                    OPENAI_API_KEY  # from secrets
                )
                df = list_to_df(list_final)
                if df.empty:
                    st.warning("No solicitations returned with the current filters.")
                else:
                    st.session_state.sol_df = df
                    st.success(f"Fetched {len(df)} solicitations.")
            except Exception as e:
                st.exception(e)

    with col2:
        if uploaded_sol is not None:
            try:
                df = pd.read_csv(uploaded_sol)
                st.session_state.sol_df = df
                st.success(f"Loaded {len(df)} solicitations from uploaded CSV.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if st.session_state.sol_df is not None:
        st.subheader("Solicitations")
        st.dataframe(st.session_state.sol_df, use_container_width=True)
        st.download_button("Download as CSV", st.session_state.sol_df.to_csv(index=False).encode("utf-8"), file_name="sol_list.csv", mime="text/csv")

# ---------------- Tab 2 ----------------
with tab2:
    st.header("Find Supplier Suggestions")
    st.write("This uses your solicitation rows + Google results (via SerpAPI) to propose suppliers and rough quotes.")
    our_rec = st.text_input("Favored suppliers (comma-separated)", value="")
    our_not = st.text_input("Do-not-use suppliers (comma-separated)", value="")
    max_google = st.number_input("Max Google results per item", min_value=1, max_value=20, value=5)

    uploaded_sup = st.file_uploader("Or upload an existing supplier_suggestions.csv", type=["csv"], key="sup_upload")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Run supplier suggestion", type="primary"):
            if st.session_state.sol_df is None:
                st.error("Load or fetch solicitations in Tab 1 first, or upload a sol_list.csv.")
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
                        OpenAi_API_Key=OPENAI_API_KEY,   # from secrets
                        Serp_API_Key=SERP_API_KEY        # from secrets
                    )
                    sup_df = pd.DataFrame(results)
                    st.session_state.sup_df = sup_df
                    st.success(f"Generated {len(sup_df)} supplier rows.")
                except Exception as e:
                    st.exception(e)

    with col2:
        if uploaded_sup is not None:
            try:
                df = pd.read_csv(uploaded_sup)
                st.session_state.sup_df = df
                st.success(f"Loaded {len(df)} supplier suggestions from uploaded CSV.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if st.session_state.sup_df is not None:
        st.subheader("Supplier suggestions")
        st.dataframe(st.session_state.sup_df, use_container_width=True)
        st.download_button("Download as CSV", st.session_state.sup_df.to_csv(index=False).encode("utf-8"), file_name="supplier_suggestions.csv", mime="text/csv")

# ---------------- Tab 3 ----------------
with tab3:
    st.header("Generate Proposal Draft")
    st.write("Select one or more supplier-suggestion rows and generate a proposal draft using your templates.")
    # Templates paths (allow upload or use defaults)
    bid_template = st.text_input("Bid template file path (DOCX)", value="/mnt/data/BID_TEMPLATE.docx")
    solinfo_template = st.text_input("Solicitation info template (DOCX)", value="/mnt/data/SOLICITATION_INFO_TEMPLATE.docx")
    out_dir = st.text_input("Output directory", value="/mnt/data/proposals")

    uploaded_sup2 = st.file_uploader("Or upload supplier_suggestions.csv here", type=["csv"], key="sup_upload2")

    if uploaded_sup2 is not None:
        try:
            df = pd.read_csv(uploaded_sup2)
            st.session_state.sup_df = df
            st.success(f"Loaded {len(df)} supplier suggestions from upload.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.sup_df is not None:
        st.dataframe(st.session_state.sup_df, use_container_width=True)
        idxs = st.multiselect("Pick rows to draft", options=list(range(len(st.session_state.sup_df))), help="Leave empty to draft all")
        run_btn = st.button("Generate proposal(s)", type="primary")
        if run_btn:
            os.makedirs(out_dir, exist_ok=True)
            try:
                if idxs:
                    df = st.session_state.sup_df.iloc[idxs]
                else:
                    df = st.session_state.sup_df
                gp.validate_supplier_and_write_proposal(
                    df=df,
                    output_directory=out_dir,
                    Open_AI_API_Key=OPENAI_API_KEY,    # from secrets
                    BID_TEMPLATE_FILE=bid_template,
                    SOl_INFO_TEMPLATE=solinfo_template
                )
                st.success(f"Drafted proposals to {out_dir}.")
                files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir,f))]
                if files:
                    st.write("Generated files:")
                    for f in files:
                        st.write(os.path.join(out_dir,f))
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.caption("This is an MVP wrapper. When you're ready, we can replace CSVs with a database and add logins.")

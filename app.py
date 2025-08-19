
import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Import user modules
import sys
sys.path.append('/mnt/data')
import get_relevant_solicitations as gs
import find_relevant_suppliers as fs
import generate_proposal as gp

# at the top of app.py
import streamlit as st, os
APP_PW = st.secrets.get("APP_PASSWORD", "")
if APP_PW:
    pw = st.text_input("Enter access password", type="password")
    if pw != APP_PW:
        st.stop()  # don't render the app for wrong/blank password

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
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required for GPT filtering & proposal drafting")
    sam_keys_raw = st.text_input("SAM.gov API Keys (comma-separated)", value="", help="Enter one or more SAM.gov API keys")
    serp_key = st.text_input("SerpAPI Key (for Google search)", type="password")
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
            if not openai_key:
                st.error("Please provide your OpenAI API key in the sidebar.")
            elif not sam_keys_raw.strip():
                st.error("Please provide at least one SAM.gov API key in the sidebar.")
            else:
                target_keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
                api_keys = [k.strip() for k in sam_keys_raw.split(",") if k.strip()]
                try:
                    list_final = gs.get_relevant_solicitation_list(days_back, int(limit_results), api_keys, target_keywords, openai_key)
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
            elif not openai_key or not serp_key:
                st.error("Please provide OpenAI & SerpAPI keys in the sidebar.")
            else:
                sol_dicts = st.session_state.sol_df.to_dict(orient="records")
                favored = [x.strip() for x in our_rec.split(",") if x.strip()]
                not_favored = [x.strip() for x in our_not.split(",") if x.strip()]
                try:
                    results = fs.get_suppliers(sol_dicts, favored, not_favored, int(max_google), openai_key, serp_key)
                    # get_suppliers likely returns a list of dicts
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
            if not openai_key:
                st.error("OpenAI API key required.")
            else:
                os.makedirs(out_dir, exist_ok=True)
                try:
                    if idxs:
                        df = st.session_state.sup_df.iloc[idxs]
                    else:
                        df = st.session_state.sup_df
                    gp.validate_supplier_and_write_proposal(df, out_dir, openai_key, bid_template, solinfo_template)
                    st.success(f"Drafted proposals to {out_dir}.")
                    # List created files
                    files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir,f))]
                    if files:
                        st.write("Generated files:")
                        for f in files:
                            st.write(os.path.join(out_dir,f))
                except Exception as e:
                    st.exception(e)

st.markdown("---")
st.caption("This is an MVP wrapper. When you're ready, we can replace CSVs with a database and add logins.")

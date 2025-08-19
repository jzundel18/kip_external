
# GovContract Assistant MVP (Streamlit)

## Run locally
1) Create a virtualenv and install deps:
   pip install -r requirements.txt

2) Launch the app:
   streamlit run app.py

3) In the sidebar, paste your API keys:
   - OpenAI API key
   - SAM.gov API key(s) (comma-separated if you have more than one)
   - SerpAPI key

## Notes
- Tab 1 fetches solicitations from SAM.gov (or load an existing CSV).
- Tab 2 generates supplier suggestions (or load an existing CSV).
- Tab 3 drafts proposals using your templates and supplier rows.

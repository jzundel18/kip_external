import requests
import openai
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
import pandas as pd
from http import HTTPStatus
from requests.exceptions import HTTPError
import json
import re

# SAM.gov Search Details
SAM_API_ENDPOINT = "https://api.sam.gov/opportunities/v2/search"

key_choice = 0 #List has two options, starting with first one.

# === Step 1: Search SAM.gov ===
def fetch_sam_solicitations(days_back, sample_size, api_keys):
    key_choice = 0
    posted_to = datetime.now(timezone.utc).date()
    posted_from = posted_to - timedelta(days=days_back)

    posted_to_str = posted_to.strftime("%m/%d/%Y")
    posted_from_str = posted_from.strftime("%m/%d/%Y")

    print(f"🔍 Querying most recent soliciations from SAM.gov")
    try:
        params = {
            "api_key": api_keys[key_choice],
            "postedFrom": posted_from_str,
            "postedTo": posted_to_str,
            "sort": "date",
            "limit": sample_size
        }
        response = requests.get(SAM_API_ENDPOINT, params=params)
        response.raise_for_status()
    except HTTPError as exc:
        if response.status_code == 404:
            print("⚠️ Error: Failed to Receive Response from SAM.gov")
            return []

        if response.status_code == 429:
            print("⚠️ Error: Usage rate exceeded on API key")
            if key_choice == 0:
                print("🔄 Switching API Key")
                key_choice = 1
            try:
                params = {
                    "api_key": api_keys[key_choice],
                    "postedFrom": posted_from_str,
                    "postedTo": posted_to_str,
                    "sort": "date",
                    "limit": sample_size
                }
                response = requests.get(SAM_API_ENDPOINT, params=params)
                response.raise_for_status()
            except HTTPError as exc:
                if response.status_code == 404:
                    print("⚠️ Error: Failed to Receive Response from SAM.gov")
                    return []
                if response.status_code == 429:
                    print("⚠️ Error: Usage rate exceeded on API keys, exiting...")
                    return []

    data = response.json()
    return data.get("opportunitiesData", [])


#=====Sept 2: Build Solicitation Library with Descriptions=======
def build_solicitation_list(solicitations, api_keys):
    sol_list = []
    key_choice = 0
    for sol in solicitations:
        notice_id = sol.get("noticeId")
        notice_type = sol.get("type")
        number = sol.get("solicitationNumber")
        naics_code = sol.get("naicsCode")
        posted_date = sol.get("postedDate")
        due_date = sol.get("responseDeadLine")
        title = sol.get("title", "No title")
        pocs = sol.get("pointOfContact")

        if pocs is not None:
            primary_contact = next((contact for contact in pocs if contact['type'] == 'primary'), None)
            if primary_contact:
                poc_email = primary_contact['email']
                poc_name = primary_contact['fullName']

            else:
                poc_email = "None"
                poc_name = "None"
        else:
            poc_email = "None"
            poc_name = "None"
        ui_link = sol.get("uiLink")

        if 'award' not in notice_type.lower():
            description_found = True
            try:
                description_url = f"https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid={notice_id}&api_key={api_keys[key_choice]}"
                response = requests.get(description_url)
                response.raise_for_status()
            except HTTPError as exc:
                if response.status_code == 404:
                    description_found = False
                    continue
                if response.status_code == 429:
                    if key_choice == 0:
                        print("🔄 Switching API Key")
                        key_choice = 1
                        try:
                            description_url = f"https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid={notice_id}&api_key={api_keys[key_choice]}"
                            response = requests.get(description_url)
                            response.raise_for_status()
                        except HTTPError as exc:
                            print("⚠ Error: Usage rate exceeded on API keys...Moving On...")
                            description_found = False
                            continue

            if description_found:
                description_html = response.text
                soup = BeautifulSoup(description_html, "html.parser")
                description = soup.get_text()
            else:
                description = "None Available"


            sol_list.append({
                "notice id": notice_id,
                "notice type": notice_type,
                "solicitation number": number,
                "NAICS Code": naics_code,
                "posted date": posted_date,
                "due date": due_date,
                "title": title,
                "point of contact name": poc_name,
                "point of contact email": poc_email,
                "description": description,
                "link": ui_link
            })
    return sol_list

# === Step 3: Filter by Keyword in Page Text ===
def filter_solicitations_by_keyword(sol_list, keywords):
    filtered_list = []
    print(f"\n🔍 Filtering {len(sol_list)} solicitations for keywords: \"{keywords}\"...\n")

    for word in keywords:
        matched = []
        for sol in sol_list:
            notice_id = sol.get("notice id")
            notice_type = sol.get("notice type")
            number = sol.get("solicitation number")
            naics_code = sol.get("NAICS Code")
            posted_date = sol.get("posted date")
            due_date = sol.get("due date")
            title = sol.get("title", "No title")
            poc_name = sol.get("point of contact name")
            poc_email = sol.get("point of contact email")
            ui_link = sol.get("link")
            description = sol.get("description")
        
            if word.lower() in description.lower():
                matched.append({
                    "notice id": notice_id,
                    "notice type": notice_type,
                    "solicitation number": number,
                    "NAICS Code": naics_code,
                    "posted date": posted_date,
                    "due date": due_date,
                    "title": title,
                    "point of contact name": poc_name,
                    "point of contact email": poc_email,
                    "description": description,
                    "link": ui_link
                })
        sol_list = matched
    filtered_list = matched
        
    return filtered_list

def extract_json(text):
    # Remove optional markdown code fences if present
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

# === Step 4: Use GPT-4 to assess and summarize each solicitation ===
def gpt_filter_solicitations(solicitations, target_keywords, OpenAi_Api_Key):
    client = openai.OpenAI(api_key=OpenAi_Api_Key)
    filtered = []
    list_final = []
    print(f"\n🔍 Using AI to filter {len(solicitations)} solicitations.")
    for sol in solicitations:
        sol_text = f"""
Title: {sol.get("title")}
NAICS Code: {sol.get("NAICS Code")}
Due Date: {sol.get("due date")}
Description: {sol.get("description", "")}
Solicitation #: {sol.get("solicitation number")}
URL: {sol.get("link")}
        """

        prompt = f"""
You are a government contracting assistant. Evaluate if this SAM.gov solicitation is a good fit.

Only select active solicitations that not past the due date. The solicitation should specifically ask for a quote or RFQ, and the solicitation should be able to be fulfilled via a US domestic supplier.

We are specifically looking for items that are goods and not services. The solicitation description should contain enough information that the product can be sourced online. The quantity required and detailed item description
must be included.

Provide the following information obtained from the solicitation description below including:
- Relevance (yes/no). The solicitation must meet the above requirements in order for a "yes" to be entered in this field. Be critical when determining relevance.
- Detailed item description including any listed required specifications. If the solicitation description mentions an attached document that contains more infomration, mention that here. Information about the related item category that can be learned from the NAICS code should also be added here.
- Quantity of items requested
- Item Part Number if mentioned in the solicitation information below. This includes NSN numbers, part numbers, product numbers, etc. Be sure to include the entire name or number
- Item Supplier if mentioned in the solicitation information below. Be sure to include the entire name or number.
- Detailed submission instructions if available including any requirments for proposal submission.
- Detailed fulfillment instructions. Include any product shipping deadlines, location where product should be shipped, and shipping address if available.

Respond ONLY with raw JSON — do not wrap it in quotes or formatting code blocks. Do not include any commentary or explanations.

Requested format:
{{
  "Relevance": "yes",
  "Item Description": "Widget X for automotive repair that must match specifications Y and Z. More details can be found in document attached to the solicitation.",
  "Quanity": "1500 units",
  "Item Part #": "WX-1234",
  "Listed Supplier": "Acme Corp",
  "Submission Instructions": "Submit your proposal via email to bids@agency.gov by July 30, 2025 at 5 PM ET.",
  "Fulfilment Instructions": "Items should be shipped to the DHS office at 1234 Walnut Ave. Newark, NJ 12345 by Sep 25, 2025.
}}
Here is the solicitation description:

{sol_text}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            result_raw = response.choices[0].message.content.strip()
            result = extract_json(result_raw)
            try:
                result_AI=json.loads(result)
                if "yes" in result.lower():
                    if result_AI.get("Quantity") != "":
                        filtered.append(result)
                        list_final.append({
                            "relevance": result_AI.get("Relevance"),
                            "notice id": sol.get("notice id"),
                            "notice type": sol.get("notice type"),
                            "solicitation number": sol.get("solicitation number"),
                            "NAICS Code": sol.get("NAICS Code"),
                            "posted date": sol.get("posted date"),
                            "due date": sol.get("due date"),
                            "title": sol.get("title"),
                            "poc name": sol.get("point of contact name"),
                            "poc email": sol.get("point of contact email"),
                            "item description": result_AI.get("Item Description"),
                            "quantity": result_AI.get("Quantity"),
                            "item part#": result_AI.get("Item Part #"),
                            "listed supplier": result_AI.get("Listed Supplier"),
                            "submission instructions": result_AI.get("Submission Instructions"),
                            "fulfillment instructions": result_AI.get("Fulfilment Instructions"),
                            "solicitation link": sol.get("link")
                        })
            except json.JSONDecodeError as e:
                print("⚠ Invalid JSON:", e)
                list_final = []
                filtered = []
        except Exception as e:
            print(f"⚠️ GPT-3.5 error: {e}")
    print(f"✅  Matched {len(list_final)} solicitations to Kenai Defense.")
    return list_final, filtered

# ====Save List====
def save_solicitation_list(list_final,filename):
    if len(list_final) != 0:
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(list_final[0:], columns=list_final[0])
        df.to_csv(f"{filename}.csv", index=False)
        print(f"Results Saved {len(list_final)} solicitations to sol_list.csv")

# === Main Runner ===
def get_relevant_solicitation_list(Days_back, N_SAM_results, Api_Keys, target_keywords, OpenAi_API_Key):
    raw_solicitations = fetch_sam_solicitations(Days_back, N_SAM_results, Api_Keys)
    solicitations = build_solicitation_list(raw_solicitations, Api_Keys)
    relevant_sols = filter_solicitations_by_keyword(solicitations, target_keywords)
    list_final, filtered_summaries = gpt_filter_solicitations(relevant_sols, target_keywords, OpenAi_API_Key)
    return list_final

# =====Run Program =====
if __name__ == "__main__":
    
    # === Configuration ===
    # API Keys
    OPENAI_API_KEY = "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"
    #SAM_API_KEY = "2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq"  # Mine
    #SAM_API_KEY = "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"    #Jayden's
    API_KEYS = ["2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq", "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"]

    # Search and Filter Parameters
    KEYWORDS = ["rfq"]
    DAYS_BACK = 30
    LIMIT_RESULTS = 50

    # Output File Name
    CSV_OUTPUT = "sol_list"
    
    relevant_sols = get_relevant_solicitation_list(DAYS_BACK, LIMIT_RESULTS, API_KEYS, KEYWORDS, OPENAI_API_KEY)
    save_solicitation_list(relevant_sols, CSV_OUTPUT)

import pandas as pd
import requests
import openai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from dateutil import parser
import re

#Today's Date
today = datetime.today()


#===== Step 1: Get Search Query from Solicitation Information =======
def generate_search_query(description, part_number, suggested_supplier, OpenAi_API_Key):
    # Load environment variables
    load_dotenv()
    openai.api_key = os.getenv("{OpenAi_API_Key}")

    client = openai.OpenAI(api_key=OpenAi_API_Key)
    
    part_number_prompt = f"The recommended part number for this solicitation is {part_number}."
    if part_number.lower() == "not listed" or part_number.lower() == "not specified" or part_number.strip().lower() == "nan":
        part_number_prompt = ""
    suggested_supplier_prompt = f"The recommended manufacturer or supplier for this solicitation is {suggested_supplier}."
    if suggested_supplier.lower() == "not listed" or suggested_supplier.lower() == "not specified" or suggested_supplier.strip().lower() == "nan":
        suggested_supplier_prompt = ""

    prompt = f"""You are a procurement assistant. Create a concise Google search query to find suppliers for the following government item description:

        Item Description:
        {description}
        {part_number_prompt}
        {suggested_supplier_prompt}

        The output should contain only the prompt which can be feed directly into a Google search box. It should not contain any additional information or explanation.
        Be sure to include any item specifications that would be required to find the correct item using a google search. Again, return only the search query.    
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = ""
    return result
#===== Step 2: Search Google using SerpAPI ======
def search_google_with_serpapi(query, serpapi_key, max_results):
    """Lightweight Google Web Search via SerpAPI using requests."""
    import requests
    results = []
    if not query or not serpapi_key:
        return results

    base_url = "https://serpapi.com/search.json"
    start = 0
    per_page = 10  # SerpAPI returns up to 10 organic results per page

    try:
        while len(results) < max_results:
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "num": per_page,
                "start": start,
            }
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            organic = data.get("organic_results", []) or []
            if not organic:
                break

            for item in organic:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })
                if len(results) >= max_results:
                    break

            start += per_page
            if not data.get("serpapi_pagination") or not data["serpapi_pagination"].get("next"):
                break
    except Exception:
        pass

    return results

# ====== Step 3: Generate AI Prompt to find suppliers ========
def generate_prompt(solicitation_desc, quantity, part_number, suggested_supplier, search_results, favored_suppliers, not_favored_suppliers):
    quantity_prompt = f"The solicitation asks for a quantity of {quantity} of the specified item."
    if quantity.lower() == "not listed" or quantity.lower() == "not specified" or quantity.strip().lower() == "nan":
        quantity_prompt = "Quantity information was not provided in the solicitation."

    part_number_prompt = f"The government customer recommended part number {part_number} for this solicitation."
    if part_number.lower() == "not listed" or part_number.lower() == "not specified" or part_number.strip().lower() == "nan":
        part_number_prompt = "The solicitation did not specify a part number or sku for this item."

    suggested_supplier_prompt = f"The government customer recommended {suggested_supplier} as the manufacturer or supplier for this solicitation."
    if suggested_supplier.lower() == "not listed" or suggested_supplier.lower() == "not specified" or suggested_supplier.strip().lower() == "nan":
        suggested_supplier_prompt = "The solicitation did not specify a manufacturer or supplier for this item."
        
    prompt =  f"""
        You are an AI sourcing assistant. Given the following government item description and a JSON list of internet search results, select the top 3 suppliers.


        Description of the product obtained from a solicitation on SAM.gov:
        ---
        {solicitation_desc}
        ---
        {quantity_prompt}
        ---
        {part_number_prompt}
        ---
        {suggested_supplier_prompt}
        ---

        JSON dictionalry containing the internet Search Results:
        {json.dumps(search_results[:5], indent=2)}

        Using only information from the internet search results, provide the **top 3 suppliers** of this item. Skip any results in the internet search that are advertisements.
        Suppliers should not be websites that are simply posting the solicitation. Ensure the suppliers actually sell parts like those listed in the solicitation description.
        Examples of websites that should not be suppliers but just post the solicitation are {', '.join(not_favored_suppliers)}, etc. It is preferred that we focus on large
        suppliers that offer bulk quantities such as {', '.join(favored_suppliers)}, etc. If none of these are in the internet search results find the best options that are avaiable.

        For each, include:
        - Relevance (yes/no)
        - Supplier Rank (1-5). Determine a ranking of each supplier based on how well the product from that supplier fits the solicitation description, where 5 is a perfect fit and 1 indicates that it is a poor fit.
        - Supplier Company Name
        - Product, item, or SKU number for the relevant product from the company's website. This may differ from suggested part number in the solicitation as long as it is an equivalent part.
        - Product name from the suppliers website. Again, this may differ from the product name in the SAM.gov solicitation. It should match the product name from the supplier website.
        - Unit price of each item from the manufacturors website.
        - Product Page URL. Make sure this is a link to the product page specifically, rather than a link to the homepage of the manufacturers website. This should come directly from the internet search results JSON dictionary above.


        Respond ONLY with raw JSON — do not wrap it in quotes or formatting code blocks. Return only valid JSON with double-escaped backslashes and escaped internal quotes where needed.
        Do not include any commentary or explanations.

        Example format:

        [
          {{
            "relevance": "yes",
            "rank": "4",
            "supplier": "Supplier 1 Name",
            "product #": "WX-1234",
            "product name": "Product Name From Supplier 1",
            "price": "$100.00",
            "url": "product page url from supplier 1 website"
          }},
          {{
            "relevance": "yes",
            "rank": "3",
            "supplier": "Supplier 2 Name",
            "product #": "WX-1234",
            "product name": "Product Name From Supplier 2",
            "price": "$100.00",
            "url": "product page url from supplier 2 website"
          }},
          {{
            "relevance": "yes",
            "rank": "5",
            "supplier": "Supplier 3 Name",
            "product #": "WX-1234",
            "product name": "Product Name From Supplier 3",
            "price": "$100.00",
            "url": "product page url from supplier 3 website"
          }}
        ]
        If no relevant suppliers were found a "no" should be output for the "relevance" category. All ofther information you include in the output must come directly from the internet
        search results json dictionary. Do not alter any item descriptions, URLs, product numbers, prices, or any other details.  If any of the required information is not included in
        the search result dictionary, output a "N/A" value.
    """
    return prompt

def import_solicitations(csv_filename):
    df = pd.read_csv(f"{csv_filename}.csv")
    sol_list = df.to_dict(orient="records")
    return sol_list

def import_suppliers(Output_CSV):
    df = pd.read_csv(Output_CSV)
    sup_list = df.to_dict(orient="records")
    return sup_list

def save_suppliers(supplier_list, Output_CSV):
    df = pd.DataFrame(supplier_list[0:], columns=supplier_list[0])
    df.to_csv(f"{Output_CSV}.csv", index=False)
    print(f"Output Supplier List to {OUTPUT_CSV}.csv")

def extract_json(text):
    # Remove optional markdown code fences if present
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

# ======== Run Program =======
def get_suppliers(solicitations, our_recommended_suppliers, our_not_recommended_suppliers, Max_Google_Results, OpenAi_API_Key, Serp_API_Key):
   
    output_with_quotes = []
    for row in solicitations:
        title = str(row.get("title", "")).strip()
        desc = str(row.get("item description", "")).strip()
        quantity = str(row.get("quantity", "")).strip()
        due_date_str = row.get("due date")
        due_date = parser.parse(due_date_str)
        if not desc:
            continue
        if quantity.lower() == "not listed" or quantity.lower() == "not specified" or quantity.strip().lower() == "nan":
            continue
        
        part = str(row.get("item part#", "")).strip()
        supplier = str(row.get("listed supplier", "")).strip()
        query = generate_search_query(desc, part, supplier, OpenAi_API_Key)

        if not query:
            continue

        search_results = search_google_with_serpapi(query, Serp_API_Key, max_results=Max_Google_Results)
        if not search_results:
            continue

        prompt = generate_prompt(desc, quantity, part, supplier, search_results, our_recommended_suppliers, our_not_recommended_suppliers)

        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("{OpenAi_API_Key}")

        client = openai.OpenAI(api_key=OpenAi_API_Key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            result_raw = response.choices[0].message.content.strip()
            result = extract_json(result_raw)
        except Exception as e:
            print(f"⚠️ GPT API error: {e}")
            result = [
              {
                "relevance": "no",
                "rank": "1",
                "supplier": "None",
                "product #": "N/A",
                "product name": "N/A",
                "price": "N/A",
                "url": "N/A"
              },
              {
                "relevance": "no",
                "rank": "1",
                "supplier": "None",
                "product #": "N/A",
                "product name": "N/A",
                "price": "N/A",
                "url": "N/A"
              },
              {
                "relevance": "no",
                "rank": "1",
                "supplier": "None",
                "product #": "N/A",
                "product name": "N/A",
                "price": "N/A",
                "url": "N/A"
              }
            ]
            result_AI=json.loads(result)
            continue
        try:
            result_AI=json.loads(result)
        except json.JSONDecodeError as e:
            print("⚠ Invalid JSON:", e)
            continue

        no_supplier ={
            "relevance": "no",
            "rank": "1",
            "supplier": "None",
            "product #": "N/A",
            "product name": "N/A",
            "price": "N/A",
            "url": "N/A"
        }

        try:
            supplier1 = result_AI[0]            
        except Exception as e:
#            print(f"⚠️ GPT API error: {e}")
            supplier1 = no_supplier
            
        try:            
            supplier2 = result_AI[1]
        except Exception as e:
#           print(f"⚠️ GPT API error: {e}")
            supplier2 = no_supplier
            
        try:
            supplier3 = result_AI[2]
        except Exception as e:
#            print(f"⚠️ GPT API error: {e}")
            supplier3 = no_supplier

        relevant = [supplier1.get("relevance"), supplier2.get("relevance"),supplier3.get("relevance")]
        if "yes" in relevant:
            if due_date.date() < today.date():
                print(f"❌ Due Date Has Passed")
                continue
            output_with_quotes.append({
                "relevance": "yes",
                "notice id": row.get("notice id"),
                "notice type": row.get("notice type"),
                "solicitation number": row.get("solicitation number"),
                "NAICS Code": row.get("NAICS Code"),
                "posted date": row.get("posted date"),
                "due date": row.get("due date"),
                "title": row.get("title"),
                "poc name": row.get("poc name"),
                "poc email": row.get("poc email"),
                "item description": row.get("item description"),
                "item part#": row.get("item part #"),
                "quantity": row.get("quantity"),
                "listed supplier": row.get("listed supplier"),
                "submission instructions": row.get("submission instructions"),
                "fulfillment instructions": row.get("fulfillment instructions"),
                "solicitation link": row.get("solicitation link"),
                "supplier 1": supplier1.get("supplier"),
                "supplier 1 rank": supplier1.get("rank"),
                "supplier 1 product #": supplier1.get("product #"),
                "supplier 1 product name": supplier1.get("product name"),
                "supplier 1 link": supplier1.get("url"),
                "supplier 1 quote": supplier1.get("price"),
                "supplier 2": supplier2.get("supplier"),
                "supplier 2 rank": supplier2.get("rank"),
                "supplier 2 product #": supplier2.get("product #"),
                "supplier 2 product name": supplier2.get("product name"),
                "supplier 2 link": supplier2.get("url"),
                "supplier 2 quote": supplier2.get("price"),
                "supplier 3": supplier3.get("supplier"),
                "supplier 3 rank": supplier3.get("rank"),
                "supplier 3 product #": supplier3.get("product #"),
                "supplier 3 product name": supplier3.get("product name"),
                "supplier 3 link": supplier3.get("url"),
                "supplier 3 quote": supplier3.get("price")
            })

    return output_with_quotes

if __name__ == "__main__":

    #File Names
    CSV_INPUT = "sol_list"
    OUTPUT_CSV = "supplier_suggestions"

    #Supplier Constraints
    RECOMMENDED_SUPPLIERS = ["Grainger", "Fastenal", "MSC Industrial Supply", "Zoro", "HD Supply", "Uline", "Ferguson", "Acme Tools", "ThermoFisher", "Newegg Business", "Digi-Key", "Mouser Electronics"]  # You can customize this list
    NOT_RECOMMENDED_SUPPLIERS = ["SAM.gov", "GovTribe", "HigherGov", "FedScout"]
    # Google Search Parameters
    MAX_RESULTS = 50 #should be a multiple of 10 up to 100. 

    sol_list = import_solicitations(CSV_INPUT)
    supplier_list = get_suppliers(sol_list, RECOMMENDED_SUPPLIERS, NOT_RECOMMENDED_SUPPLIERS, MAX_RESULTS, OPENAI_API_KEY, serpapi_api_key)
    save_suppliers(supplier_list, OUTPUT_CSV)

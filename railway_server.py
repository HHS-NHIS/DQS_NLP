# DQS_SERVER2_Railway.py
# Railway deployment version with enhanced CORS and port handling

import os, re, json, unicodedata, urllib.parse, csv
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import httpx

def _dqs_slug(s: str) -> str:
    s = (s or '').lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s

# Global variables to hold loaded keyword data
LOADED_KEYWORDS = {}
DATASET_METADATA = {}
TOPIC_ROUTES = {}
DQS_TOPIC_MAPPING = {}
GROUPING_CATEGORIES = {}

APP_NAME = "cdc_health_data_system"
APP_VERSION = "1.0.0-railway"
PAGE_LIMIT = int(os.getenv("SOCRATA_PAGE_LIMIT", "5000"))
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
HEADERS = {"X-App-Token": SOCRATA_APP_TOKEN} if SOCRATA_APP_TOKEN else {}
CAT_PATH = os.getenv("DQS_CATALOG_PATH", "dqs_catalog.csv")
USER_KEYWORDS_PATH = os.getenv("DQS_KEYWORDS_PATH", "dqs_keywords.json")
DQS_ALLOWED_DATASETS = os.getenv("DQS_ALLOWED_DATASETS", "").strip()

app = FastAPI(title="CDC Health Data Question System")

# Enhanced CORS for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your work domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOCRATA_API_FIELDS = [
    "topic","subtopic","taxonomy","classification",
    "group","group_id","subgroup","subgroup_id",
    "time_period","time_period_id","estimate","standard_error",
    "estimate_lci","estimate_uci","flag","footnote_id_list"
]

def norm(s: str) -> str:
    return re.sub(r"\s+"," ", unicodedata.normalize("NFKC", str(s or "")).strip().lower())

def tokenize(s: str): 
    return re.findall(r"[a-z0-9]+", norm(s))

async def fetch(url: str):
    try:
        async with httpx.AsyncClient(timeout=90, headers=HEADERS) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []

def create_enhanced_error_message(query: str) -> str:
    main_topic = extract_main_health_topic(query)
    cdc_link = get_relevant_cdc_link(query)
    
    if main_topic:
        return f"Estimates for **{main_topic}** are not available in our data query system at this time, but more information can be found at the CDC: [Click here for {main_topic} information]({cdc_link}). You can also try asking about available topics like diabetes, heart disease, high blood pressure, asthma, depression, flu vaccination, or cancer."
    else:
        return f"The requested health information is not available in our data query system at this time, but you may find relevant information at: [CDC Health Information]({cdc_link}). Try asking about available topics like diabetes, heart disease, high blood pressure, asthma, depression, flu vaccination, or cancer."

def extract_main_health_topic(query: str) -> str:
    if not query:
        return ""
    
    query_lower = query.lower().strip()
    remove_phrases = [
        "how many", "what is", "what are", "tell me about", "information about",
        "data on", "statistics for", "prevalence of", "rates of", "cases of",
        "people with", "adults with", "children with", "by gender", "by race",
        "by age", "by state", "in men", "in women", "in adults", "in children"
    ]
    
    for phrase in remove_phrases:
        query_lower = query_lower.replace(phrase, "")
    
    words = query_lower.split()
    stop_words = {"the", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"}
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if words:
        return " ".join(words[:3])
    
    return query.strip()

def get_relevant_cdc_link(query: str) -> str:
    if not query:
        return "https://www.cdc.gov/"
    
    query_lower = query.lower()
    
    CDC_TOPIC_LINKS = {
        "suicide": "https://www.cdc.gov/suicide/index.html",
        "dental": "https://www.cdc.gov/oralhealth/index.html",
        "diabetes": "https://www.cdc.gov/diabetes/index.html",
        "cancer": "https://www.cdc.gov/cancer/index.htm"
    }
    
    for topic, link in CDC_TOPIC_LINKS.items():
        if topic in query_lower:
            return link
    
    if any(term in query_lower for term in ["vaccine", "vaccination", "immunization", "shot"]):
        return "https://www.cdc.gov/vaccines/index.html"
    elif any(term in query_lower for term in ["mental", "depression", "anxiety", "suicide"]):
        return "https://www.cdc.gov/mentalhealth/index.htm"
    else:
        return "https://www.cdc.gov/"

def get_dqs_topic_slug(indicator: str, query: str = "", dataset_info: dict = None) -> str:
    if not indicator:
        return ""
    
    indicator_clean = indicator.lower().strip()
    query_clean = (query or "").lower().strip()
    
    # Special handling for drug overdose - needs topic/subtopic structure
    if "drug overdose" in indicator_clean or "overdose" in indicator_clean:
        return "drug-overdose-deaths"
    
    # Special handling for suicide
    if "suicide" in indicator_clean:
        return "suicide"
    
    # Special handling for healthcare spending topics
    if any(term in indicator_clean for term in ["spending", "expenditure", "cost"]):
        if "personal healthcare" in indicator_clean:
            return "personal-healthcare-spending"
        elif "national health" in indicator_clean:
            return "national-health-expenditures"
        elif "physician" in indicator_clean or "clinical services" in indicator_clean:
            return "physician-and-clinical-services"
        elif "hospital care" in indicator_clean:
            return "hospital-care"
        elif "dental services" in indicator_clean:
            return "dental-services"
        elif "prescription" in indicator_clean and "drug" in indicator_clean:
            return "prescription-drugs"
        elif "nursing care" in indicator_clean:
            return "nursing-care-facilities"
        elif "home health" in indicator_clean:
            return "home-health-care"
        else:
            return "personal-healthcare-spending"  # Default for spending queries
    
    # Enhanced DQS mapping lookup
    if DQS_TOPIC_MAPPING:
        # First try exact matches
        for key, value in DQS_TOPIC_MAPPING.items():
            if key.lower() == indicator_clean:
                return value
        
        # Then try substring matches
        for key, value in DQS_TOPIC_MAPPING.items():
            if key.lower() in indicator_clean or indicator_clean in key.lower():
                return value
    
    children_keywords = ["children", "child", "kids", "pediatric", "childhood", "youth"]
    is_children_context = any(kw in indicator_clean for kw in children_keywords) or any(kw in query_clean for kw in children_keywords)
    
    if any(term in indicator_clean for term in ["flu", "influenza"]):
        if is_children_context:
            return "receipt-of-influenza-vaccination-among-children"
        else:
            return "receipt-of-influenza-vaccination-among-adults"
    
    # Enhanced topic detection for common health topics
    if "dental" in indicator_clean or "oral" in indicator_clean:
        if "caries" in indicator_clean or "decay" in indicator_clean:
            return "total-dental-caries-in-permanent-teeth-in-adults"
        elif "loss" in indicator_clean:
            return "complete-tooth-loss"
        else:
            return "dental-exam-or-cleaning"
    
    return _dqs_slug(indicator_clean)

def get_dqs_group_slug(group: str) -> str:
    if not group:
        return ""
    
    group_clean = group.lower().strip()
    
    if GROUPING_CATEGORIES:
        for category_name, category_data in GROUPING_CATEGORIES.items():
            for key, value in category_data.items():
                if key in group_clean or group_clean in key:
                    return _dqs_slug(value)
    
    return _dqs_slug(group_clean)

def load_user_keywords():
    global LOADED_KEYWORDS, DATASET_METADATA, TOPIC_ROUTES, DQS_TOPIC_MAPPING, GROUPING_CATEGORIES
    
    print("📄 Loading keywords from dqs_keywords.json...")
    
    if os.path.exists(USER_KEYWORDS_PATH):
        try:
            with open(USER_KEYWORDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Load all sections
            LOADED_KEYWORDS = data.get("topic_keywords", {})
            DATASET_METADATA = data.get("dataset_metadata", {})
            TOPIC_ROUTES = data.get("topic_routes", {})
            DQS_TOPIC_MAPPING = data.get("dqs_topic_mapping", {})
            GROUPING_CATEGORIES = data.get("grouping_categories", {})
            
            print(f"✅ Successfully loaded:")
            print(f"   - {len(LOADED_KEYWORDS)} topic keywords")
            print(f"   - {len(DATASET_METADATA)} dataset entries")
            print(f"   - {len(TOPIC_ROUTES)} topic routes")
            print(f"   - {len(DQS_TOPIC_MAPPING)} DQS mappings")
            print(f"   - {len(GROUPING_CATEGORIES)} grouping categories")
            
            # Sample some loaded data for verification
            print(f"📊 Sample keywords: {list(LOADED_KEYWORDS.keys())[:5]}")
            print(f"📊 Sample topics with routes: {list(TOPIC_ROUTES.keys())[:5]}")
            
        except Exception as e:
            print(f"❌ Error loading keywords: {e}")
            # Initialize empty dictionaries as fallback
            LOADED_KEYWORDS = {}
            DATASET_METADATA = {}
            TOPIC_ROUTES = {}
            DQS_TOPIC_MAPPING = {}
            GROUPING_CATEGORIES = {}
    else:
        print(f"❌ Keywords file not found: {USER_KEYWORDS_PATH}")
        print("❌ All keyword matching will fail!")
        # Initialize empty dictionaries
        LOADED_KEYWORDS = {}
        DATASET_METADATA = {}
        TOPIC_ROUTES = {}
        DQS_TOPIC_MAPPING = {}
        GROUPING_CATEGORIES = {}

def extract_socrata_id_from_url(url: str) -> Optional[str]:
    pats = [r'/resource/([a-z0-9\-]+)\.json', r'/resource/([a-z0-9\-]+)/?$', r'/d/([a-z0-9\-]+)/?']
    for p in pats:
        m = re.search(p, url or "")
        if m: 
            return m.group(1)
    return None

async def discover_dataset_structure(domain: str, dsid: str):
    structure = {
        "raw_fields": SOCRATA_API_FIELDS, 
        "mapped_fields": {}, 
        "available_values": {}, 
        "sample_data": [], 
        "discovery_log": [], 
        "dqs_like": False
    }
    
    try:
        sample_url = f"https://{domain}/resource/{dsid}.json?$select=topic,`group`,subgroup,time_period,estimate,estimate_lci,estimate_uci&$where=topic IS NOT NULL AND estimate IS NOT NULL&$limit=10"
        sample_data = await fetch(sample_url)
        structure["sample_data"] = sample_data
        
        if not sample_data:
            sample_url = f"https://{domain}/resource/{dsid}.json?$limit=10"
            sample_data = await fetch(sample_url)
            structure["sample_data"] = sample_data
        
        keys = set()
        for r in sample_data:
            if isinstance(r,dict): 
                keys.update([k.lower() for k in r.keys()])
        
        required = {"topic","group","subgroup","time_period","estimate"}
        structure["dqs_like"] = required.issubset(keys)
        structure["mapped_fields"] = {
            "indicator":"topic",
            "grouping_category":"group",
            "group_value":"subgroup",
            "year":"time_period",
            "value":"estimate"
        }

        av = {}
        for logical, api_field in [("indicator","topic"),("grouping_category","group"),("group_value","subgroup"),("year","time_period")]:
            try:
                field = api_field if api_field!="group" else f"`{api_field}`"
                urls_to_try = [
                    f"https://{domain}/resource/{dsid}.json?$select={field}&$group={field}&$limit=1000",
                    f"https://{domain}/resource/{dsid}.json?$select={field}&$group={field}&$limit=500"
                ]
                
                vals = []
                for url in urls_to_try:
                    rows = await fetch(url)
                    if rows:
                        new_vals = [row.get(api_field) for row in rows if row.get(api_field)]
                        vals.extend(new_vals)
                        if len(vals) > 10:
                            break
                
                av[logical] = sorted(set(str(v) for v in vals if v))
                
            except Exception as e:
                print(f"⚠️ Error fetching {logical} for {dsid}: {e}")
                av[logical] = []
        
        structure["available_values"] = av
        structure["dqs_like"] = bool(structure["dqs_like"]) and bool(av.get("indicator")) and bool(av.get("group_value"))
        
        # Special logging for key datasets
        key_datasets = ["w26f-tf3h", "36ue-xht5", "h3hw-hzvg", "rdjz-vn2n", "7aq9-prdf"]
        if dsid in key_datasets:
            print(f"📊 {dsid}: DQS-like={structure['dqs_like']}, indicators={len(av.get('indicator', []))}")
    
    except Exception as e:
        print(f"⚠️ Error discovering structure for {dsid}: {e}")
        structure["dqs_like"] = False
    
    return structure

async def load_and_map_catalog():
    print("📚 Loading catalog...")
    
    # First load keywords
    load_user_keywords()
    
    catalog = {}
    
    if DATASET_METADATA:
        print("📄 Loading from dqs_keywords.json")
        for dsid, metadata in DATASET_METADATA.items():
            if metadata.get("domain") == "data.cdc.gov":
                catalog[dsid] = {
                    "domain": metadata["domain"],
                    "label": metadata["name"],
                    "api_url": f"https://data.cdc.gov/resource/{dsid}.json",
                    "description": metadata.get("description", metadata["name"]),
                    "target_population": [metadata.get("population", "general")],
                    "age_range": metadata.get("age_range", "all ages"),
                    "data_source": metadata.get("data_source", "CDC"),
                    "topics": metadata.get("topics", []),
                    "structure": None,
                    "dqs_like": False
                }
    
    if os.path.exists(CAT_PATH):
        try:
            with open(CAT_PATH,"r",encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = None
                    label = None
                    for col,val in row.items():
                        if val and ('http' in val or 'data.cdc.gov' in val): 
                            url = val.strip()
                        elif val and not label and len(val.strip())>3: 
                            label = val.strip()
                    if url and label:
                        dsid = extract_socrata_id_from_url(url)
                        if dsid and dsid not in catalog:
                            catalog[dsid] = {
                                "domain":"data.cdc.gov",
                                "label": label,
                                "api_url": f"https://data.cdc.gov/resource/{dsid}.json",
                                "description": label,
                                "target_population": (["adults","18+","adult"] if "adult" in label.lower() else (["children","kids","child"] if "child" in label.lower() else ["general"])),
                                "structure": None,
                                "dqs_like": False
                            }
        except Exception as e:
            print(f"⚠️ Could not load CSV catalog: {e}")

    print(f"📊 Total datasets to check: {len(catalog)}")

    for dsid, info in list(catalog.items()):
        struct = await discover_dataset_structure(info["domain"], dsid)
        info["structure"] = struct
        info["dqs_like"] = bool(struct.get("dqs_like"))
        inds = struct.get("available_values",{}).get("indicator",[]) or []
        info["available_indicators"] = inds

    filtered = {}
    for k,v in catalog.items():
        struct = v.get("structure",{})
        av = struct.get("available_values",{})
        if v.get("dqs_like") and av.get("indicator") and av.get("group_value"):
            filtered[k]=v
    catalog = filtered

    allow = [x.strip() for x in DQS_ALLOWED_DATASETS.split(",") if x.strip()]
    if allow:
        catalog = {k:v for k,v in catalog.items() if k in allow}

    print(f"✅ Final catalog size: {len(catalog)} DQS-compatible datasets")
    return catalog

def find_canonical_topic(query_text: str) -> str:
    if not LOADED_KEYWORDS:
        print("❌ No keywords loaded - cannot find canonical topic")
        return ""
    
    query_lower = query_text.lower().strip()
    
    # Exact word matching first
    query_words = query_lower.split()
    for word in query_words:
        if word in LOADED_KEYWORDS:
            canonical = LOADED_KEYWORDS[word]
            print(f"🎯 Found exact word: '{word}' -> '{canonical}'")
            return canonical
    
    # Substring matching
    best_match = ""
    longest_match = 0
    
    for keyword, canonical in LOADED_KEYWORDS.items():
        if keyword in query_lower:
            if len(keyword) > longest_match:
                best_match = canonical
                longest_match = len(keyword)
    
    if best_match:
        print(f"🎯 Found substring: '{best_match}'")
        return best_match
    
    # Token overlap matching
    query_tokens = set(tokenize(query_text))
    best_score = 0
    
    for keyword, canonical in LOADED_KEYWORDS.items():
        keyword_tokens = set(tokenize(keyword))
        overlap = len(query_tokens & keyword_tokens)
        if overlap > best_score and overlap > 0:
            best_score = overlap
            best_match = canonical
    
    if best_match:
        print(f"🎯 Found token overlap: '{best_match}'")
    
    return best_match

def get_relevant_datasets_for_topic(canonical_topic: str) -> List[str]:
    if not TOPIC_ROUTES or not canonical_topic:
        return []
    
    topic_data = TOPIC_ROUTES.get(canonical_topic, {})
    if isinstance(topic_data, dict):
        dataset_ids = list(topic_data.values())
        print(f"📊 Found datasets for '{canonical_topic}': {dataset_ids}")
        return dataset_ids
    
    return []

def _indicator_best_with_score(query: str, available_indicators: List[str]):
    if not available_indicators: 
        return (None, -1)
    
    ql = query.lower()
    qtokens = set(tokenize(query))
    best, score_best = None, -1
    
    children_keywords = ["children", "child", "kids", "pediatric", "childhood", "youth"]
    is_children_query = any(kw in ql for kw in children_keywords)
    
    canonical_topic = find_canonical_topic(query)
    
    # Enhanced keyword matching for specific topics
    dental_terms = ["dental", "teeth", "tooth", "caries", "oral", "dentist", "periodontal", "gum", "cleaning", "exam"]
    suicide_terms = ["suicide", "suicidal", "self-harm", "self harm", "suicide rate", "death", "mortality"]
    flu_terms = ["flu shot", "flu vaccine", "influenza vaccine", "influenza vaccination", "seasonal flu", "flu", "influenza"]
    cancer_terms = ["cancer", "malignant", "tumor", "breast", "lung", "prostate", "cervical", "skin"]
    heart_terms = ["heart", "cardiac", "coronary", "cardiovascular", "heart attack", "angina"]
    overdose_terms = ["overdose", "drug overdose", "opioid", "heroin", "fentanyl"]
    spending_terms = ["spending", "cost", "costs", "expenditure", "expenditures", "physician services", "hospital care", "dental services", "prescription drugs", "nursing care", "home health"]
    
    is_dental_query = any(term in ql for term in dental_terms)
    is_suicide_query = any(term in ql for term in suicide_terms)
    is_cancer_query = any(term in ql for term in cancer_terms)
    is_heart_query = any(term in ql for term in heart_terms)
    is_overdose_query = any(term in ql for term in overdose_terms)
    is_spending_query = any(term in ql for term in spending_terms)
    
    for ind in available_indicators:
        il = ind.lower()
        s = 0
        
        is_children_indicator = any(kw in il for kw in children_keywords)
        
        # Topic-specific scoring
        if is_dental_query:
            if any(term in il for term in dental_terms):
                s += 60
                if "caries" in ql and "caries" in il:
                    s += 40
                if "dental" in ql and "dental" in il:
                    s += 30
                if "tooth" in ql and "tooth" in il:
                    s += 25
        
        elif is_suicide_query:
            if any(term in il for term in suicide_terms):
                s += 60
                if "suicide" in ql and "suicide" in il:
                    s += 40
                if "death" in ql and "death" in il:
                    s += 30
        
        elif is_cancer_query:
            if any(term in il for term in cancer_terms):
                s += 50
                if "cancer" in ql and "cancer" in il:
                    s += 30
        
        elif is_heart_query:
            if any(term in il for term in heart_terms):
                s += 50
                if "heart" in ql and "heart" in il:
                    s += 30
        
        elif is_overdose_query:
            if any(term in il for term in overdose_terms):
                s += 60
                if "overdose" in ql and "overdose" in il:
                    s += 40
        
        elif is_spending_query:
            if any(term in il for term in spending_terms):
                s += 50
                if "spending" in ql and "spending" in il:
                    s += 30
                # Specific spending type bonuses
                if "physician" in ql and "physician" in il:
                    s += 25
                elif "hospital care" in ql and "hospital" in il:
                    s += 25
                elif "dental services" in ql and "dental" in il:
                    s += 25
                elif "prescription" in ql and "prescription" in il:
                    s += 25
        
        elif canonical_topic and canonical_topic.lower() in il:
            s += 50
        
        elif any(term in ql for term in flu_terms):
            if any(term in il for term in ["influenza", "flu"]) and "pneumo" not in il:
                s += 40
                if any(vacc_term in il for vacc_term in ["vaccin", "immun", "shot"]):
                    s += 20
                if is_children_query and is_children_indicator:
                    s += 25
                elif is_children_query and not is_children_indicator:
                    s -= 15
        
        # Keyword mapping bonus
        if LOADED_KEYWORDS:
            for keyword, canonical in LOADED_KEYWORDS.items():
                if keyword in ql and canonical.lower() in il:
                    s += 25
        
        # Token matching (excluding topic-specific cases)
        if not any([is_dental_query, is_suicide_query, is_cancer_query, is_heart_query, is_overdose_query, is_spending_query]):
            for t in qtokens:
                if len(t)>=4 and t in il: 
                    s += 5
            if ql in il: 
                s += 8
        
        # Population matching
        if is_children_query and is_children_indicator:
            s += 15
        elif is_children_query and not is_children_indicator:
            s -= 10
        
        if s > score_best:
            best, score_best = ind, s
    
    return best, score_best

def select_best_dataset(query: str, catalog: Dict[str, Any]):
    if not catalog: 
        return None
    
    ql = query.lower()
    
    children_keywords = ["children", "child", "kids", "pediatric", "childhood", "youth"]
    is_children_query = any(kw in ql for kw in children_keywords)
    
    canonical_topic = find_canonical_topic(query)
    if canonical_topic:
        relevant_dataset_ids = get_relevant_datasets_for_topic(canonical_topic)
        if relevant_dataset_ids:
            filtered_catalog = {dsid: info for dsid, info in catalog.items() if dsid in relevant_dataset_ids}
            if filtered_catalog:
                catalog = filtered_catalog
                print(f"🎯 Filtered to {len(catalog)} relevant datasets for '{canonical_topic}'")
    
    qtokens = set(tokenize(query))
    scores = {}
    
    # Enhanced topic detection
    vaccination_terms = ["flu", "influenza", "vaccine", "vaccination", "shot", "immunization"]
    dental_terms = ["dental", "teeth", "tooth", "caries", "oral", "dentist"]
    suicide_terms = ["suicide", "suicidal", "self-harm", "self harm", "death"]
    cancer_terms = ["cancer", "malignant", "tumor", "breast", "lung", "prostate"]
    heart_terms = ["heart", "cardiac", "coronary", "cardiovascular"]
    overdose_terms = ["overdose", "drug overdose", "opioid", "heroin"]
    spending_terms = ["spending", "cost", "costs", "expenditure", "expenditures", "healthcare spending", "medical costs"]
    
    is_vaccination_query = any(term in ql for term in vaccination_terms)
    is_dental_query = any(term in ql for term in dental_terms)
    is_suicide_query = any(term in ql for term in suicide_terms)
    is_cancer_query = any(term in ql for term in cancer_terms)
    is_heart_query = any(term in ql for term in heart_terms)
    is_overdose_query = any(term in ql for term in overdose_terms)
    is_spending_query = any(term in ql for term in spending_terms)
    
    for dsid, info in catalog.items():
        inds = info.get("available_indicators",[]) or []
        label_lower = info.get("label","").lower()
        topics = info.get("topics", [])
        
        topic_relevance = 0
        
        # Canonical topic bonus
        if canonical_topic and dsid in get_relevant_datasets_for_topic(canonical_topic):
            topic_relevance += 70
            
            # Specialist dataset bonus
            if any(specialist in label_lower for specialist in ["oral health", "suicide", "mental health", "cardiovascular", "nutrition"]):
                topic_relevance += 50
            elif any(general in label_lower for general in ["summary health", "chronic disease indicators"]):
                topic_relevance += 15
        
        # Topic-specific dataset matching
        if is_dental_query:
            if "oral health" in label_lower or "dental" in label_lower:
                topic_relevance += 80
            elif any(term in topics for term in ["oral health", "dental care", "dental caries"]):
                topic_relevance += 70
            elif "summary health" in label_lower:
                topic_relevance -= 20
                
        elif is_suicide_query:
            if "suicide" in label_lower or "mortality" in label_lower or "vital statistics" in label_lower:
                topic_relevance += 90
            elif any(term in topics for term in ["suicide", "mortality", "death rates"]):
                topic_relevance += 80
            elif "summary health" in label_lower:
                topic_relevance -= 30
        
        elif is_cancer_query:
            if "cancer" in label_lower or "mortality" in label_lower:
                topic_relevance += 80
            elif any(term in topics for term in ["cancer", "cancer mortality"]):
                topic_relevance += 70
        
        elif is_heart_query:
            if "heart" in label_lower or "cardiovascular" in label_lower:
                topic_relevance += 80
            elif any(term in topics for term in ["heart disease", "cardiovascular"]):
                topic_relevance += 70
        
        elif is_overdose_query:
            if "drug" in label_lower or "overdose" in label_lower or "mortality" in label_lower:
                topic_relevance += 80
            elif any(term in topics for term in ["drug overdose", "opioids", "substance use"]):
                topic_relevance += 70
        
        elif is_spending_query:
            if any(term in label_lower for term in ["spending", "expenditure", "cost", "healthcare spending"]):
                topic_relevance += 90
                
                # Specific spending type bonuses
                if "personal healthcare" in ql and "personal" in label_lower:
                    topic_relevance += 30
                elif "national" in ql and "national" in label_lower:
                    topic_relevance += 30
                elif "physician" in ql and any(term in topics for term in ["physician services", "clinical services"]):
                    topic_relevance += 20
                elif "hospital care" in ql and "hospital" in topics:
                    topic_relevance += 20
                elif "dental services" in ql and "dental" in topics:
                    topic_relevance += 20
                elif "prescription" in ql and "prescription" in topics:
                    topic_relevance += 20
                    
            elif any(term in topics for term in ["healthcare spending", "health expenditures", "spending"]):
                topic_relevance += 70
        
        elif is_vaccination_query:
            vaccination_indicators = [ind for ind in inds if any(term in ind.lower() for term in ["flu", "influenza", "vaccine", "vaccination", "immunization"])]
            if vaccination_indicators:
                topic_relevance += 60
                if is_children_query:
                    children_vacc_indicators = [ind for ind in vaccination_indicators if any(kw in ind.lower() for kw in children_keywords)]
                    if children_vacc_indicators:
                        topic_relevance += 30
            elif any(term in label_lower for term in vaccination_terms):
                topic_relevance += 40
            else:
                topic_relevance -= 20
        
        # Indicator matching
        indicator_match = 0
        topic_specific_indicators = []
        
        if is_dental_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in dental_terms)]
        elif is_suicide_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in suicide_terms)]
        elif is_cancer_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in cancer_terms)]
        elif is_heart_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in heart_terms)]
        elif is_overdose_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in overdose_terms)]
        elif is_spending_query:
            topic_specific_indicators = [ind for ind in inds if any(term in ind.lower() for term in spending_terms)]
        
        if topic_specific_indicators:
            indicator_match += 40
        else:
            # General indicator matching
            for ind in inds:
                il = str(ind).lower()
                if il in ql or ql in il: 
                    indicator_match = max(indicator_match, 20)
                elif any(t for t in qtokens if len(t)>=4 and t in il): 
                    indicator_match = max(indicator_match, 15)
        
        # Population matching
        population_match = 0
        if is_children_query:
            if any(x in label_lower for x in ["child", "children", "kids", "pediatric", "youth"]):
                population_match += 20
            elif any(x in label_lower for x in ["adult", "adults", "18+"]):
                population_match -= 15
        else:
            if any(x in ql for x in ["adult","adults","18+"]): 
                if "adult" in label_lower: 
                    population_match += 15
            elif not is_children_query:
                if "adult" in label_lower: 
                    population_match += 8
        
        final_score = topic_relevance + indicator_match + population_match
        scores[dsid] = final_score
        
        if dsid in ["w26f-tf3h", "36ue-xht5", "h3hw-hzvg", "rdjz-vn2n", "s57w-7gbe", "gu48-2cs8"]:
            print(f"📊 {dsid}: relevance={topic_relevance}, indicator={indicator_match}, pop={population_match}, final={final_score}")
    
    if not scores:
        return None
        
    best_dsid = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_dsid]
    print(f"🏆 Best dataset: {best_dsid} (score: {best_score})")
    return best_dsid

def detect_grouping(query: str, structure: Dict[str, Any]):
    av = structure.get("available_values",{})
    groups = av.get("grouping_category",[]) or []
    subgroups = av.get("group_value",[]) or []
    ql = query.lower()
    
    if GROUPING_CATEGORIES:
        for category_name, category_data in GROUPING_CATEGORIES.items():
            for key, value in category_data.items():
                if key in ql:
                    g = next((x for x in groups if value.lower() in x.lower() or x.lower() in value.lower()), None)
                    if g:
                        return g, None, False
    
    GROUPING_KEYWORDS = {
        "sex": "Sex", "gender":"Sex",
        "male": ("Sex","Male"), "female": ("Sex","Female"),
        "men": ("Sex","Male"), "women": ("Sex","Female"),
        "race": "Race and Hispanic origin group",
        "ethnicity": "Race and Hispanic origin group",
        "origin": "Race and Hispanic origin group",
        "age": "Age group", "age group":"Age group",
        "education": "Education", "income":"Income", "poverty":"Federal poverty level",
        "state": "State or territory", "region":"Region",
        "urban":"Urbanization level","metro":"Urbanization level",
        "veteran":"Veteran status","marital":"Marital status","insurance":"Health insurance coverage"
    }
    
    for kw, mapping in GROUPING_KEYWORDS.items():
        if kw in ql:
            if isinstance(mapping, tuple):
                g_name, sg_name = mapping
                g = next((x for x in groups if g_name.lower() in x.lower()), None)
                sg = next((x for x in subgroups if sg_name.lower() in x.lower()), None)
                return g, sg, True
            else:
                g = next((x for x in groups if mapping.lower() in x.lower() or x.lower() in mapping.lower()), None)
                if g: 
                    return g, None, False
    
    for sg in subgroups:
        toks=set(tokenize(sg))
        if toks and toks.issubset(set(tokenize(query))):
            g = None
            for cand in groups:
                gl=cand.lower()
                if any(k in gl for k in ["sex","gender"]) and any(w in ql for w in ["male","female","men","women"]): 
                    g=cand
                    break
                if any(k in gl for k in ["race","hispanic","origin","ethnic"]) and any(w in ql for w in ["white","black","asian","hispanic","latino"]): 
                    g=cand
                    break
                if "state" in gl and "state" in ql: 
                    g=cand
                    break
            return g, sg, True
    return None, None, False

def detect_year(query: str):
    m = re.search(r"\b(20\d{2}|19\d{2})\b", query or "")
    if m: 
        return m.group(1)
    return None

def detect_two_groupings(query: str, structure: Dict[str, Any]) -> list:
    q = (query or "").lower()
    av = (structure or {}).get("available_values", {}) if structure else {}
    avail = [str(g) for g in (av.get("grouping_category", []) or [])]

    synonym_sets = {
        "race": ["race", "hispanic", "ethnicity", "origin"],
        "sex": ["sex", "gender", "male", "female", "men", "women", "sex at birth"],
        "age": ["age", "ages", "age group", "age groups"],
        "education": ["education", "educational", "attainment"],
        "state or territory": ["state", "territory", "state or territory"],
        "geographic characteristic": ["geographic", "geography", "region", "regions", "urban", "rural", "metro", "nonmetropolitan"],
    }
    hits = []
    for canon, kws in synonym_sets.items():
        pos = None
        for kw in kws:
            i = q.find(kw)
            if i != -1:
                pos = i if pos is None else min(pos, i)
        if pos is not None:
            label = None
            cl = canon.lower()
            for g in avail:
                gl = g.lower()
                if cl in gl or gl in cl:
                    label = g
                    break
            if label:
                hits.append((pos, label))
    hits.sort(key=lambda t: t[0])
    result = []
    for _, label in hits:
        if label not in result:
            result.append(label)
    return result[:2]

def build_dqs_query_url(domain: str, dsid: str, **filters):
    where = []
    if filters.get("indicator"):
        val = str(filters["indicator"]).replace("'", "''")
        where.append(f"topic='{val}'")
    if filters.get("grouping_category"):
        val = str(filters["grouping_category"]).replace("'", "''")
        where.append(f"`group`='{val}'")
    if filters.get("group_value"):
        val = str(filters["group_value"]).replace("'", "''")
        where.append(f"subgroup='{val}'")
    if filters.get("year"):
        val = str(filters["year"]).replace("'", "''")
        where.append(f"time_period='{val}'")
    
    select = ",".join([
        "topic","`group`","subgroup","time_period","estimate","standard_error","estimate_lci","estimate_uci","flag","footnote_id_list"
    ])
    
    # Increase limit for mortality datasets to get all years
    limit = PAGE_LIMIT
    if dsid in ["w26f-tf3h", "rdjz-vn2n", "h3hw-hzvg", "7aq9-prdf"]:  # mortality datasets
        limit = 10000
    
    params = {"$select": select, "$limit": str(limit), "$order":"time_period DESC, subgroup"}
    if where: 
        params["$where"]=" AND ".join(where)
    return f"https://{domain}/resource/{dsid}.json?" + urllib.parse.urlencode(params)

async def get_catalog():
    global CATALOG
    if not hasattr(get_catalog, 'CATALOG') or not get_catalog.CATALOG:
        get_catalog.CATALOG = await load_and_map_catalog()
    return get_catalog.CATALOG

async def process_nlq_query(query: str):
    catalog = await get_catalog()
    
    requested_groups = []
    grp_unavailable = []
    
    if not catalog: 
        return {
            "error": "Sorry, we couldn't find any health data to search. This might mean the data files aren't set up correctly. Please contact support or try again later.", 
            "requested_groups": requested_groups, 
            "unavailable_requested_groups": grp_unavailable
        }

    canonical_topic = find_canonical_topic(query)
    print(f"🔍 Query: '{query}' -> Canonical topic: '{canonical_topic}'")

    dsid = select_best_dataset(query, catalog)
    info = catalog.get(dsid)
    if not info: 
        error_msg = "Sorry, we couldn't find a health dataset that matches your question."
        
        if canonical_topic:
            relevant_datasets = get_relevant_datasets_for_topic(canonical_topic)
            if relevant_datasets:
                missing_datasets = [ds for ds in relevant_datasets if ds not in catalog]
                if missing_datasets:
                    error_msg += f" We found that '{canonical_topic}' data should be in datasets {missing_datasets}, but these datasets aren't currently available or compatible."
                else:
                    error_msg += f" We found the topic '{canonical_topic}' but couldn't match it to available datasets."
            else:
                error_msg += f" We identified the topic as '{canonical_topic}' but no datasets are configured for this topic."
        else:
            error_msg += " We couldn't identify a specific health topic from your question."
        
        error_msg += " Try asking about common health topics like diabetes, heart disease, or mental health."
        
        return {
            "error": error_msg,
            "canonical_topic": canonical_topic,
            "loaded_keywords_count": len(LOADED_KEYWORDS),
            "available_datasets": list(catalog.keys()),
            "relevant_datasets": get_relevant_datasets_for_topic(canonical_topic) if canonical_topic else []
        }
    
    structure = info["structure"]
    available_indicators = info.get("available_indicators", [])

    best_ind, best_score = _indicator_best_with_score(query, available_indicators)
    print(f"🎯 Best indicator: '{best_ind}' (score: {best_score})")
    
    dataset_content_issue = False
    if canonical_topic:
        expected_terms = []
        if canonical_topic == "suicide":
            expected_terms = ["suicide", "death", "mortality", "injury"]
        elif canonical_topic == "dental care" or canonical_topic == "oral health":
            expected_terms = ["dental", "oral", "teeth", "tooth", "caries"]
        elif canonical_topic == "cancer":
            expected_terms = ["cancer", "malignant", "tumor", "breast", "lung", "prostate"]
        elif canonical_topic == "heart disease":
            expected_terms = ["heart", "cardiac", "coronary", "cardiovascular"]
        elif canonical_topic == "drug overdose":
            expected_terms = ["overdose", "drug", "opioid", "heroin"]
        
        if expected_terms:
            matching_indicators = [ind for ind in available_indicators if any(term in ind.lower() for term in expected_terms)]
            non_matching_indicators = [ind for ind in available_indicators if not any(term in ind.lower() for term in expected_terms)]
            
            if not matching_indicators and non_matching_indicators:
                dataset_content_issue = True
                print(f"⚠️ DATASET CONTENT MISMATCH: Expected {canonical_topic} indicators but found: {non_matching_indicators[:3]}")
    
    # Lowered threshold for better matching
    indicator = best_ind if (best_score is not None and best_score >= 8) else None

    tmp = detect_grouping(query, structure)
    if isinstance(tmp, tuple) and len(tmp) == 3: 
        group, subgroup, subgroup_locked = tmp
    else: 
        group, subgroup, subgroup_locked = tmp[0], tmp[1], bool(tmp[1])

    year = detect_year(query)

    filters = {}
    if indicator is None:
        available_inds = info.get("available_indicators", [])
        
        if dataset_content_issue:
            error_msg = f"**Dataset Content Issue**: We found the '{info.get('label')}' dataset for '{canonical_topic}', but it contains different data than expected."
            error_msg += f" The dataset has indicators like '{available_inds[0] if available_inds else 'None'}' instead of {canonical_topic}-related data."
            error_msg += f" This might be a data configuration issue. Please try a different search term or contact support."
            
            other_datasets = [ds for ds in catalog.keys() if ds != dsid][:3]
            if other_datasets:
                error_msg += f" You might also try searching in other available datasets: {', '.join(other_datasets)}."
        else:
            error_msg = create_enhanced_error_message(query)
            
            if canonical_topic:
                error_msg += f" We found your topic ('{canonical_topic}') and selected the '{info.get('label')}' dataset, but couldn't find a matching indicator."
                if available_inds:
                    similar_inds = [ind for ind in available_inds[:5] if any(word in ind.lower() for word in query.lower().split() if len(word) > 3)]
                    if similar_inds:
                        error_msg += f" Similar indicators available: {', '.join(similar_inds[:3])}"
        
        return {
            "query": query, 
            "dataset_id": dsid, 
            "dataset_info": info, 
            "structure": structure,
            "matches": {
                "indicator": None, 
                "grouping_category": None, 
                "group_value": None, 
                "year": year, 
                "subgroup_locked": False
            },
            "filters_used": {}, 
            "query_url": build_dqs_query_url(info["domain"], dsid),
            "data_count": 0, 
            "data": [],
            "error": error_msg,
            "canonical_topic": canonical_topic,
            "available_indicators": available_inds[:10],
            "best_score": best_score,
            "dataset_content_issue": dataset_content_issue
        }

    if indicator: 
        filters["indicator"]=indicator
    if group: 
        filters["grouping_category"]=group
    if subgroup: 
        filters["group_value"]=subgroup
    if year: 
        filters["year"]=year

    requested_groups = detect_two_groupings(query, structure)
    avail_groups = (structure or {}).get("available_values", {}).get("grouping_category", []) or []
    grp_unavailable = []
    if requested_groups:
        for rg in requested_groups:
            found = any(rg.lower() in str(ag).lower() for ag in avail_groups)
            if not found:
                grp_unavailable.append(rg)

    url = build_dqs_query_url(info["domain"], dsid, **filters)
    data = await fetch(url)

    return {
        "query": query,
        "dataset_id": dsid,
        "dataset_info": info,
        "structure": structure,
        "matches": {
            "indicator": indicator, 
            "grouping_category": group, 
            "group_value": subgroup, 
            "year": year, 
            "subgroup_locked": subgroup_locked,
            "requested_groups": requested_groups,
            "unavailable_requested_groups": grp_unavailable
        },
        "filters_used": filters, 
        "query_url": url, 
        "data_count": len(data), 
        "data": data,
        "canonical_topic": canonical_topic
    }

def format_results_for_chart(res: Dict[str, Any]) -> Dict[str, Any]:
    data = res.get("data", []) or []
    out = dict(res)
    m0 = res.get("matches", {}) or {}

    def fnum(x):
        try:
            return float(str(x).replace("%","").replace(",",""))
        except Exception:
            return None

    if not data:
        out["chart_series"] = []
        out["chart_description"] = "No chart available."
        out["chart_summary"] = ""
        out["smart_narrative"] = "**No data found** for this query."
        return out

    if not m0.get("grouping_category"):
        _grp_total = [row for row in data if str(row.get("group","")).strip().lower() == "total"]
        if _grp_total:
            data = _grp_total
        else:
            _tot = {"total","overall","both sexes","all persons","all adults","all children","all"}
            only = [row for row in data if str(row.get("subgroup","")).strip().lower() in _tot]
            if only:
                data = only

    wanted_sg = (m0.get("group_value") or "").strip()
    if m0.get("subgroup_locked") and wanted_sg:
        data = [row for row in data if str(row.get("subgroup","")).strip().lower() == wanted_sg.lower()]

    sm = {}
    periods_set = set()
    for row in data:
        sg = str(row.get("subgroup","Overall"))
        tp = str(row.get("time_period","Unknown"))
        est, l, u = fnum(row.get("estimate")), fnum(row.get("estimate_lci")), fnum(row.get("estimate_uci"))
        if est is None: 
            continue
        periods_set.add(tp)
        sm.setdefault(sg,{})
        sm[sg][tp] = {"estimate": est, "estimate_lci": l, "estimate_uci": u}

    chart_series = []
    for sg, tdict in sm.items():
        pts = []
        for tp, vals in sorted(tdict.items(), key=lambda kv: kv[0]):
            d = {"time_period": tp, "estimate": vals["estimate"]}
            if vals.get("estimate_lci") is not None: 
                d["estimate_lci"] = vals["estimate_lci"]
            if vals.get("estimate_uci") is not None: 
                d["estimate_uci"] = vals["estimate_uci"]
            pts.append(d)
        chart_series.append({"label": sg, "points": pts})

    out["chart_series"] = chart_series

    narrative_parts = []
    indicator = m0.get("indicator") or "the selected indicator"
    group = m0.get("grouping_category") or "overall"
    query = res.get("query", "")
    
    dqs_base = "https://nchsdata.cdc.gov/DQS/"
    topic_slug = get_dqs_topic_slug(indicator, query, res.get("dataset_info"))
    group_slug = get_dqs_group_slug(group) if group and group != "overall" else ""
    year_param = str(m0.get("year") or "")
    
    # Special handling for drug overdose and suicide URLs
    subtopic_param = ""
    if "drug overdose" in indicator.lower() or "overdose" in indicator.lower():
        if "all drug overdose" in indicator.lower():
            subtopic_param = "all-drug-overdose-deaths"
        elif "opioid" in indicator.lower():
            subtopic_param = "drug-overdose-deaths-involving-any-opioid"
        elif "heroin" in indicator.lower():
            subtopic_param = "drug-overdose-deaths-involving-heroin"
        else:
            subtopic_param = "all-drug-overdose-deaths"
    elif "suicide" in indicator.lower():
        subtopic_param = "suicide-among-adults-aged-geq-18-years"
    elif "physician and clinical services" in indicator.lower() or "physician" in indicator.lower():
        subtopic_param = "physician-and-clinical-services"
    elif "hospital care" in indicator.lower():
        subtopic_param = "hospital-care"  
    elif "dental services" in indicator.lower():
        subtopic_param = "dental-services"
    elif "prescription drugs" in indicator.lower() and "spending" in query.lower():
        subtopic_param = "prescription-drugs"
    elif "nursing care" in indicator.lower():
        subtopic_param = "nursing-care-facilities"
    elif "home health" in indicator.lower():
        subtopic_param = "home-health-care"
    elif "personal healthcare" in indicator.lower() and "spending" in indicator.lower():
        subtopic_param = "personal-healthcare-spending"
    elif "national health" in indicator.lower() and ("expenditure" in indicator.lower() or "spending" in indicator.lower()):
        subtopic_param = "national-health-expenditures"
    
    dqs_url = f"{dqs_base}?topic={topic_slug}&subtopic={subtopic_param}&group={group_slug}&subgroup=&range={year_param}"
    out["chart_dqs_url"] = dqs_url
    
    def fmt_ci(a, b):
        try:
            if a is not None and b is not None:
                return f" (95% CI: {a:.1f}%-{b:.1f}%)"
        except:
            pass
        return ""
    
    requested_groups = m0.get("requested_groups") or []
    if len(requested_groups) >= 2:
        first, second = requested_groups[0], requested_groups[1]
        shown = group
        narrative_parts.append(f"**Note:** You asked for data by {first.title()} and {second.title()}. Our health data shows one breakdown at a time, so we're currently showing data by **{shown}**.")
        
        btns = []
        if first != shown.lower():
            btns.append(f"<button onclick=\"setQuery('{indicator} by {first}');ask()\" class='switch-btn'>Show by {first.title()}</button>")
        if second != shown.lower():
            btns.append(f"<button onclick=\"setQuery('{indicator} by {second}');ask()\" class='switch-btn'>Show by {second.title()}</button>")
        
        if btns:
            narrative_parts.append("<div class='switch-options'>" + " ".join(btns) + "</div>")

    requested_year = m0.get("year")
    periods = sorted(list(periods_set))
    
    if requested_year and len(periods) > 1:
        narrative_parts.append(f"**About the Data:** You asked for {requested_year} data, but this dataset has information from multiple years ({', '.join(periods)}). The chart shows trends across all available years so you can see how things have changed over time.")
    
    grp_unavailable = m0.get("unavailable_requested_groups") or []
    if not group or group == "overall":
        if grp_unavailable:
            narrative_parts.append(f"**Data Note:** You asked for information by **{grp_unavailable[0]}**, but this dataset doesn't break down the data that way. Instead, we're showing you the overall numbers for everyone.")
        elif " by " in query.lower() and not requested_groups:
            parts = query.lower().split(" by ")
            if len(parts) > 1:
                requested_demo = parts[-1].strip()
                narrative_parts.append(f"**Data Note:** You asked for information by **{requested_demo}**, but this dataset doesn't break down the data that way. Instead, we're showing you the overall numbers for everyone.")
        else:
            narrative_parts.append(f"**What you're seeing:** Overall numbers for **{indicator}** across the entire population.")
    else:
        year_text = requested_year or ("all available years" if len(periods) > 1 else (periods[0] if periods else ""))
        narrative_parts.append(f"**What you're seeing:** {indicator} broken down by **{group}** for {year_text}.")

    if len(chart_series) > 1:
        try:
            all_estimates = []
            for s in chart_series:
                for pt in s.get("points", []):
                    if pt.get("estimate") is not None:
                        all_estimates.append({
                            "group": s.get("label", "Unknown"),
                            "year": pt.get("time_period", "Unknown"), 
                            "estimate": pt["estimate"],
                            "lci": pt.get("estimate_lci"),
                            "uci": pt.get("estimate_uci")
                        })
            
            if all_estimates:
                highest = max(all_estimates, key=lambda x: x["estimate"])
                lowest = min(all_estimates, key=lambda x: x["estimate"])
                narrative_parts.append(f"**Key Findings:** The highest rate was **{highest['estimate']:.1f}%** for {highest['group']} in {highest['year']}{fmt_ci(highest['lci'], highest['uci'])}. The lowest rate was **{lowest['estimate']:.1f}%** for {lowest['group']} in {lowest['year']}{fmt_ci(lowest['lci'], lowest['uci'])}.")
        except Exception:
            pass

    if wanted_sg and m0.get("subgroup_locked"):
        focus_period = requested_year or (periods[-1] if periods else "")
        focus_series = next((s for s in chart_series if (s.get("label") or "").lower() == wanted_sg.lower()), None)
        if focus_series:
            point = None
            if focus_period:
                point = next((p for p in focus_series.get("points", []) if p.get("time_period") == focus_period), None)
            if point is None and focus_series.get("points"):
                point = focus_series["points"][-1]
            if point:
                val = point.get("estimate")
                lci = point.get("estimate_lci")
                uci = point.get("estimate_uci")
                if val is not None:
                    period_label = focus_period or point.get("time_period")
                    narrative_parts.append(f"**Specific Number:** For {wanted_sg} in {period_label}: **{val:.1f}%**{fmt_ci(lci, uci)}.")

    narrative_parts.append(f"**About the Numbers:** The percentages show how common this health condition is. When you see ranges in parentheses, those show us how confident we can be in the numbers (95% confidence intervals).")
    narrative_parts.append(f"**Want More Details?** [Click here for detailed analysis tools]({dqs_url}) where you can create custom charts and get more specific breakdowns.")
    
    out["smart_narrative"] = " ".join(narrative_parts)

    year_text = (m0.get("year") or ("all available years" if len(periods)>1 else (periods[0] if periods else "")))
    desc = f"This chart shows how common {indicator} is, broken down by {group} for {year_text}. The error bars show how confident we can be in these numbers."
    desc += f" Click here for more detailed analysis tools: {dqs_url}"
    out["chart_description"] = desc
    out["chart_summary"] = ""

    return out

# HTML INTERFACE ENDPOINTS

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main landing page"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CDC Health Data Query System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            color: #333;
            text-align: center;
        }
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .subtitle {
            font-size: 1.3em;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }
        .feature {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #3498db;
            text-align: left;
            transition: transform 0.3s ease;
        }
        .feature:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        .feature h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .cta-buttons {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 40px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 15px 30px;
            font-size: 1.1em;
            font-weight: bold;
            text-decoration: none;
            border-radius: 50px;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            display: inline-block;
        }
        .btn-primary {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }
        .btn-secondary {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .examples {
            background: #ecf0f1;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: left;
        }
        .examples h3 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .example-queries {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .example-query {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
            font-style: italic;
            color: #555;
        }
        .status-section {
            background: #2ecc71;
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 30px 0;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 CDC Health Data Query System</h1>
        <p class="subtitle">Ask questions about health data in plain English and get instant, interactive visualizations</p>
        
        <div class="status-section">
            <h3>✅ System Status: Online & Ready</h3>
            <p>Railway deployment active • Enhanced topic matching • Comprehensive health data coverage</p>
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>🤖 Natural Language Queries</h3>
                <p>Ask questions like "What are suicide rates by gender?" or "Show me dental care statistics for children" and get instant results.</p>
            </div>
            <div class="feature">
                <h3>📊 Interactive Visualizations</h3>
                <p>Get beautiful charts with confidence intervals, trend lines, and smart narratives that explain what the data means.</p>
            </div>
            <div class="feature">
                <h3>🎯 Smart Topic Detection</h3>
                <p>Advanced AI automatically identifies health topics and routes to the most relevant CDC datasets for accurate results.</p>
            </div>
            <div class="feature">
                <h3>🔧 Embeddable Widget</h3>
                <p>Compact 900x300 widget perfect for embedding in websites, dashboards, or applications. Quick Q&A format with brief, accurate answers.</p>
            </div>
            <div class="feature">
                <h3>🔗 CDC Integration</h3>
                <p>Direct links to official CDC resources and detailed analysis tools for deeper research and verification.</p>
            </div>
        </div>
        
        <div class="cta-buttons">
            <a href="/nlq" class="btn btn-primary">🚀 Start Querying Data</a>
            <a href="/nlq_tester" class="btn btn-secondary">🧪 Try Example Queries</a>
            <a href="/widget" class="btn btn-primary">🔧 Compact Widget</a>
        </div>
        
        <div class="examples">
            <h3>Example Questions You Can Ask:</h3>
            <div class="example-queries">
                <div class="example-query">"What are suicide rates by state?"</div>
                <div class="example-query">"Show me dental care statistics for children"</div>
                <div class="example-query">"Cancer mortality trends by gender"</div>
                <div class="example-query">"Drug overdose deaths in 2022"</div>
                <div class="example-query">"Heart disease rates by age group"</div>
                <div class="example-query">"Flu vaccination coverage by race"</div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Powered by FastAPI • Railway Cloud • CDC Open Data</strong></p>
            <p>Version 1.0.0-railway | <a href="/health" style="color: #3498db;">Health Check</a> | <a href="/__version" style="color: #3498db;">API Info</a></p>
        </div>
    </div>
</body>
</html>
    """)

@app.get("/nlq", response_class=HTMLResponse)
async def nlq_interface():
    """Main query interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CDC Health Data Query Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px 20px 0 0;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 2.2em;
        }
        .header p {
            color: #7f8c8d;
            margin: 0;
            font-size: 1.1em;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 0 0 20px 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        .query-section {
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        .query-container {
            display: flex;
            gap: 15px;
            align-items: center;
            max-width: 800px;
            margin: 0 auto;
        }
        #queryInput {
            flex: 1;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #3498db;
            border-radius: 50px;
            outline: none;
            transition: all 0.3s ease;
        }
        #queryInput:focus {
            border-color: #2980b9;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        #askButton {
            padding: 15px 30px;
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        #askButton:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        #askButton:disabled {
            background: #bdc3c7;
            transform: none;
            cursor: not-allowed;
        }
        .quick-queries {
            margin-top: 20px;
            text-align: center;
        }
        .quick-queries h4 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        .quick-query-btn {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: #ecf0f1;
            color: #2c3e50;
            text-decoration: none;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }
        .quick-query-btn:hover {
            background: #3498db;
            color: white;
            transform: translateY(-1px);
        }
        .results-section {
            padding: 30px;
            min-height: 400px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #7f8c8d;
            font-size: 18px;
        }
        .loading::after {
            content: "";
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #e74c3c;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .success {
            background: #2ecc71;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .chart-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }
        .narrative {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
            line-height: 1.6;
        }
        .switch-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .switch-btn:hover {
            background: #c0392b;
            transform: translateY(-1px);
        }
        .switch-options {
            margin: 15px 0;
            text-align: center;
        }
        .back-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(52, 152, 219, 0.9);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        .back-btn:hover {
            background: rgba(41, 128, 185, 0.9);
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <a href="/" class="back-btn">← Back to Home</a>
    
    <div class="header">
        <h1>🔍 Health Data Query Interface</h1>
        <p>Ask questions about health data in plain English</p>
    </div>
    
    <div class="main-container">
        <div class="query-section">
            <div class="query-container">
                <input type="text" id="queryInput" placeholder="Ask a health question... (e.g., 'What are suicide rates by gender?')" />
                <button id="askButton" onclick="ask()">Ask</button>
            </div>
            
            <div class="quick-queries">
                <h4>Quick Examples:</h4>
                <button class="quick-query-btn" onclick="setQuery('suicide rates by gender')">Suicide rates by gender</button>
                <button class="quick-query-btn" onclick="setQuery('dental care for children')">Dental care for children</button>
                <button class="quick-query-btn" onclick="setQuery('cancer mortality by state')">Cancer mortality by state</button>
                <button class="quick-query-btn" onclick="setQuery('drug overdose deaths 2022')">Drug overdose deaths 2022</button>
                <button class="quick-query-btn" onclick="setQuery('heart disease by age')">Heart disease by age</button>
                <button class="quick-query-btn" onclick="setQuery('flu vaccination coverage')">Flu vaccination coverage</button>
            </div>
        </div>
        
        <div class="results-section">
            <div id="results">
                <div style="text-align: center; padding: 50px; color: #7f8c8d;">
                    <h3>🚀 Ready to explore health data!</h3>
                    <p>Enter a question above or click one of the example queries to get started.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentChart = null;

        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }

        async function ask() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const button = document.getElementById('askButton');
            const results = document.getElementById('results');
            
            // Disable button and show loading
            button.disabled = true;
            button.textContent = 'Searching...';
            results.innerHTML = '<div class="loading">Analyzing your question and searching health data</div>';

            try {
                const response = await fetch('/v1/nlq', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ q: query })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'An error occurred');
                }

                displayResults(data);
            } catch (error) {
                results.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
            } finally {
                button.disabled = false;
                button.textContent = 'Ask';
            }
        }

        function displayResults(data) {
            const results = document.getElementById('results');
            let html = '';

            // Success message
            html += `<div class="success"><strong>✅ Query successful!</strong> Found ${data.data_count} data points from CDC dataset: ${data.dataset_info.label}</div>`;

            // Chart
            if (data.chart_series && data.chart_series.length > 0) {
                html += `
                    <div class="chart-container">
                        <div class="chart-title">${data.matches.indicator || 'Health Data Visualization'}</div>
                        <canvas id="resultsChart" width="400" height="200"></canvas>
                    </div>
                `;
            }

            // Smart narrative
            if (data.smart_narrative) {
                html += `<div class="narrative">${data.smart_narrative}</div>`;
            }

            // Raw data info
            html += `
                <div class="chart-container">
                    <h4>📊 Query Details</h4>
                    <p><strong>Dataset:</strong> ${data.dataset_info.label}</p>
                    <p><strong>Indicator:</strong> ${data.matches.indicator || 'N/A'}</p>
                    <p><strong>Grouping:</strong> ${data.matches.grouping_category || 'Overall'}</p>
                    <p><strong>Data Points:</strong> ${data.data_count}</p>
                    ${data.chart_dqs_url ? `<p><strong>CDC Analysis Tools:</strong> <a href="${data.chart_dqs_url}" target="_blank">View in CDC DQS →</a></p>` : ''}
                </div>
            `;

            results.innerHTML = html;

            // Create chart if data exists
            if (data.chart_series && data.chart_series.length > 0) {
                createChart(data.chart_series);
            }
        }

        function createChart(chartSeries) {
            const ctx = document.getElementById('resultsChart').getContext('2d');
            
            // Destroy existing chart if it exists
            if (currentChart) {
                currentChart.destroy();
            }

            // Prepare datasets
            const datasets = chartSeries.map((series, index) => {
                const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
                const color = colors[index % colors.length];
                
                return {
                    label: series.label,
                    data: series.points.map(point => ({
                        x: point.time_period,
                        y: point.estimate,
                        lci: point.estimate_lci,
                        uci: point.estimate_uci
                    })),
                    borderColor: color,
                    backgroundColor: color + '20',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.1
                };
            });

            // Get all unique time periods for x-axis
            const allPeriods = [...new Set(
                chartSeries.flatMap(series => 
                    series.points.map(point => point.time_period)
                )
            )].sort();

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allPeriods,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: false
                        },
                        legend: {
                            display: chartSeries.length > 1,
                            position: 'top'
                        },
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    const dataPoint = context.raw;
                                    if (dataPoint.lci && dataPoint.uci) {
                                        return `95% CI: ${dataPoint.lci.toFixed(1)}% - ${dataPoint.uci.toFixed(1)}%`;
                                    }
                                    return '';
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time Period'
                            }
                        }
                    }
                }
            });
        }

        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                ask();
            }
        });

        // Focus on input when page loads
        document.getElementById('queryInput').focus();
    </script>
</body>
</html>
    """)

# API ENDPOINTS (keeping all existing ones)

@app.post("/v1/nlq")
async def nlq(body: Dict[str, Any] = Body(...)):
    q = str(body.get("q","")).strip()
    if not q: 
        return JSONResponse({"error":"Please enter a health question to search for."}, status_code=400)
    res = await process_nlq_query(q)
    if "error" in res: 
        return JSONResponse(res, status_code=400)
    return format_results_for_chart(res)

@app.get("/v1/keywords")
async def keywords():
    return {
        "loaded_keywords": LOADED_KEYWORDS,
        "dataset_metadata_count": len(DATASET_METADATA),
        "topic_routes_count": len(TOPIC_ROUTES),
        "dqs_topic_mapping_count": len(DQS_TOPIC_MAPPING),
        "grouping_categories": GROUPING_CATEGORIES,
        "sample_keywords": dict(list(LOADED_KEYWORDS.items())[:10]) if LOADED_KEYWORDS else {}
    }

@app.get("/__version")
def version(): 
    return {"name": APP_NAME, "version": APP_VERSION, "file": __file__}

@app.get("/__whoami")
def whoami():
    import os
    return {"pid": os.getpid(), "cwd": os.getcwd(), "file": __file__}

# Health check endpoint for Railway
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "CDC Health Data API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print("✅ CDC Health Data Question System Ready for Railway!")
    print("🔗 Ask questions in plain English about health topics")
    print("🎨 User-friendly interface with helpful error messages")
    print("📊 Easy-to-understand charts and explanations")
    print("🗂️ Uses dqs_keywords.json for enhanced matching!")
    print("🔧 COMPREHENSIVE TOPIC FIXES APPLIED!")
    print(f"🚀 Starting on port {port}")
    print("🌐 Railway deployment ready!")
    print("📄 Required: dqs_catalog.csv and dqs_keywords.json files")
    print("🌐 Access the web interface at:")
    print("   • Landing page: /")
    print("   • Main interface: /nlq")
    print("   • Example queries: /nlq_tester")
    print("   • Compact widget: /widget (900x300 iframe)")
    print("   • Widget API: /v1/widget/ask")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

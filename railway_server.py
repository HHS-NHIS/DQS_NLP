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

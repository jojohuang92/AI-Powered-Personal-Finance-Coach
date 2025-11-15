from transformers import pipeline
import re
from datetime import datetime

def extract_receipt(text: str):
    try:
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/distilbert-base-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        
        entities = ner_pipeline(text)
        
        result = process_entities(entities, text)
        
        return result
        
    except Exception as e:
        return rule_based_extraction(text)

def process_entities(entities, original_text):
    extracted_data = {
        "merchant": "",
        "amount": "",
        "date": "",
        "transaction_type": classify_transaction_type(original_text),
        "payment_method": classify_payment_method(original_text)
    }
    
    for entity in entities:
        if entity['entity_group'] == 'ORG':
            extracted_data["merchant"] += entity['word'] + " "
        elif entity['entity_group'] == 'MISC' and any(char.isdigit() for char in entity['word']):
            if '$' in entity['word'] or re.search(r'\d+\.\d{2}', entity['word']):
                extracted_data["amount"] = entity['word']
    
    extracted_data["merchant"] = extracted_data["merchant"].strip()
    
    if not extracted_data["amount"]:
        extracted_data["amount"] = extract_amount_rule_based(original_text)
    if not extracted_data["date"]:
        extracted_data["date"] = extract_date_rule_based(original_text)
    if not extracted_data["merchant"]:
        extracted_data["merchant"] = extract_merchant_rule_based(original_text)
    
    return extracted_data

def rule_based_extraction(text: str):
    return {
        "merchant": extract_merchant_rule_based(text),
        "amount": extract_amount_rule_based(text),
        "date": extract_date_rule_based(text),
        "transaction_type": classify_transaction_type(text),
        "payment_method": classify_payment_method(text)
    }

def extract_merchant_rule_based(text: str):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    url_match = re.search(r'([a-zA-Z0-9]+)\.com', text, re.IGNORECASE)
    if url_match:
        domain = url_match.group(1)
        return domain.title()
    
    for i, line in enumerate(lines):
        if re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}', line):
            for j in range(1, min(3, i+1)):
                candidate = lines[i-j]
                if is_valid_merchant_candidate(candidate):
                    return candidate
    
    skip_words = ['receipt', 'transaction', 'sale', 'copy', 'thank you', 'customer', 'date', 'time']
    for line in lines[:8]:
        line_lower = line.lower()
        if (len(line) > 3 and 
            not any(word in line_lower for word in skip_words) and
            not re.match(r'^\d', line) and
            not re.match(r'^[\d\s\.\$\%]+$', line)):
            return line
    
    return "Unknown Merchant"

def is_valid_merchant_candidate(text: str):
    """Check if text could be a valid merchant name"""
    if len(text) < 2 or len(text) > 100:
        return False
    
    skip_patterns = [
        r'^\d',
        r'^[\d\s\.\$\%]+$',
        r'receipt|transaction|sale|copy|thank|date|time|total|tax|payment|card|customer',
    ]
    
    text_lower = text.lower()
    for pattern in skip_patterns:
        if re.search(pattern, text_lower):
            return False
    
    return True

def extract_amount_rule_based(text: str):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'TOTAL' in line.upper():
            amount_match = re.search(r'[\$]?\s*(\d+\.\d{2})', line)
            if amount_match:
                return amount_match.group(1)
            if i + 1 < len(lines):
                amount_match = re.search(r'[\$]?\s*(\d+\.\d{2})', lines[i + 1])
                if amount_match:
                    return amount_match.group(1)
    for line in lines:
        if 'TOTAL PURCHASE' in line.upper():
            amount_match = re.search(r'[\$]?\s*(\d+\.\d{2})', line)
            if amount_match:
                return amount_match.group(1)
    all_amounts = re.findall(r'[\$]?\s*(\d+\.\d{2})', text)
    if all_amounts:
        amounts_float = [float(amt) for amt in all_amounts]
        return f"{max(amounts_float):.2f}"
    return None

def extract_date_rule_based(text: str):
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'(\d{1,2}-\d{1,2}-\d{2,4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return datetime.now().strftime("%m/%d/%Y")

def classify_transaction_type(text: str):
    text_lower = text.lower()
    credit_indicators = ['refund', 'return', 'credit', 'deposit', 'payment received']
    if any(keyword in text_lower for keyword in credit_indicators):
        return "credit"
    return "debit"

def classify_payment_method(text: str):
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ['visa', 'mastercard', 'amex', 'american express', 'discover', 'credit card']):
        return "credit card"
    elif any(keyword in text_lower for keyword in ['debit card', 'check card', 'bank card']):
        return "debit card"
    elif any(keyword in text_lower for keyword in ['cash', 'currency']):
        return "cash"
    elif any(keyword in text_lower for keyword in ['check', 'cheque']):
        return "check"
    else:
        return "unknown"

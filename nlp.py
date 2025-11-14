import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
#from dataset import load_dataset, ClassLabel, Sequence

def extract_receipt(text: str):
    ner_model = DistilBertForTokenClassification.from_pretrained("./ner_model")
    ner_tokenizer = DistilBertTokenizerFast.from_pretrained("./ner_model")

    cls_model = DistilBertForTokenClassification.from_pretrainted("./classifier_model")
    cls_tokenizer = DistilBertTokenizerFast.from_pretrained("./classifier_model")

    tokens = ner_tokenizer(text.split(), return_tensors="pt", is_split_into_words=True)
    outputs = ner_model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)

    entities = {"merchant": "", "amount": "", "date": ""}
    labels = [ner_model.config.id2label[i.item()] for i in predictions[0]]

    for token, label in zip(text.split(), labels):
        if "MERCHANT" in label:
            entities["merchant"] += token + " "
        elif "AMOUNT" in label:
            entities["amount"] += token + " "
        elif "DATE" in label:
            entities["date"] += token + " "

    entities = {k: v.strip() for k, v in entities.items() if v.strip()}

    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True)
    logits = cls_model(**inputs).logits
    pred_class_id = torch.argmax(logits, dim=1).item()
    transaction_type = cls_model.config.id2label[pred_class_id]

    result = {
        "merchant": entities.get("merchant", None),
        "amount": entities.get("amount", None),
        "date": entities.get("date", None),
        "transaction_type": transaction_type
    }

    return result
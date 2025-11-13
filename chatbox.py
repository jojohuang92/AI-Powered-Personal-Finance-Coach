import streamlit as st
import os
import pandas as pd
from google import genai
from dotenv import load_dotenv

def response(financial_data: pd.DataFrame, message: str):
    try:
        load_dotenv()
        api = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api)
    except AttributeError:
        st.error("API Key not found")
    
    
    data_string = financial_data.to_string()
    
    prompt = f"You are an expert financial coach. Analyze the following financial data and answer the user's question.\n\nFinancial Data:\n{data_string}\n\nUser's Question:\n{message}"
    
    ai_response = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
    return ai_response.text
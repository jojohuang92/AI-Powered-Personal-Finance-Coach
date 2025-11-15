import streamlit as st
import os
import pandas as pd
from google import genai
from dotenv import load_dotenv

def response(financial_data: pd.DataFrame, budgets: dict, message: str):
    try:
        load_dotenv()
        api = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api)
    except AttributeError:
        st.error("API Key not found")
    
    debits_df = financial_data[financial_data['Transaction Type'] == 'debit']
    credits_df = financial_data[financial_data['Transaction Type'] == 'credit']

    total_income = credits_df['Amount'].sum()
    total_expense = debits_df['Amount'].sum()
    net_savings = total_income - total_expense
    category_spending = debits_df.groupby('Category')['Amount'].sum().to_dict()

    summary_data = f"""
    - Total Income: ${total_income:,.2f}
    - Total Expenses: ${total_expense:,.2f}
    - Net Savings: ${net_savings:,.2f}
    - Spending by Category: {category_spending}
    - Budgets by Category: {budgets}
    """
    
    prompt = f"You are an expert financial coach. Analyze the following financial data and answer the user's question.\n\nFinancial Data:\n{summary_data}\n\nUser's Question:\n{message}"
    
    ai_response = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
    return ai_response.text

def analysis(financial_data: pd.DataFrame):
    try:
        load_dotenv()
        api = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api)
    except AttributeError:
        st.error("API Key not found")

    debits_df = financial_data[financial_data['Transaction Type'] == 'debit']
    credits_df = financial_data[financial_data['Transaction Type'] == 'credit']

    total_income = credits_df['Amount'].sum()
    total_expense = debits_df['Amount'].sum()
    net_savings = total_income - total_expense

    category_spending = debits_df.groupby('Category')['Amount'].sum().to_dict()

    prompt = f"You are an AI expert financial coach. Analyze the following financial data. Provide a short-term and a long-term financial goal for the user.\nFinancial Data:\n{financial_data}"
    

    analysis_response = client.models.generate_content(model= "gemini-2.5-flash", contents= prompt)
    
    return analysis_response.text

    
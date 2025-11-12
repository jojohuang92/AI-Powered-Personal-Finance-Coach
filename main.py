import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="AI Powered Personal Finance Coach", page_icon="💰", layout="wide")

category_file = "categories.json"
account_file = "accounts.json"

if "categories" not in st.session_state:
    if os.path.exists(category_file):
        with open(category_file, "r") as f:
            st.session_state.categories = json.load(f)
    else:
        st.session_state.categories = {
            "Uncategorized": []
        }

if "accounts" not in st.session_state:
    if os.path.exists(account_file):
        with open(account_file, "r") as f:
            st.session_state.accounts = json.load(f)
    else:
        st.session_state.accounts = []

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def save_accounts():
    with open(account_file, "w") as f:
        json.dump(st.session_state.accounts, f)

def load_transactions(file):
    try:
        df = pd.read_csv(file)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def transaction_form(transaction_type, df):
    st.subheader(f"Add New {transaction_type.capitalize()} Transaction")
    with st.form(f"new_{transaction_type}_transaction_form", clear_on_submit=True):
        date = st.date_input("Date")
        description = st.text_input("Description")
        
        category_options = list(st.session_state.categories.keys())
        category = st.selectbox("Category", options=category_options)
        
        amount = st.number_input("Amount", format="%.2f", min_value=0.0)

        df_accounts = []
        if 'Account Name' in df.columns:
            df_accounts = df['Account Name'].dropna().unique().tolist()
        
        all_accounts = sorted(list(set(df_accounts + st.session_state.accounts)))
        account_name = st.selectbox("Account Name", options=all_accounts)
        
        submitted = st.form_submit_button(f"Add {transaction_type.capitalize()} Transaction")
        
        if submitted:
            new_transaction = {
                "Date": pd.to_datetime(date),
                "Description": description,
                "Category": category,
                "Amount": amount,
                "Transaction Type": transaction_type,
                "Account Name": account_name
            }
            
            for col in df.columns:
                if col not in new_transaction:
                    new_transaction[col] = None
            
            new_transaction_df = pd.DataFrame([new_transaction])
            st.session_state.df = pd.concat([st.session_state.df, new_transaction_df], ignore_index=True)
            st.success("Transaction added successfully!")
            st.rerun()

def manage_accounts_ui(key_prefix):
    st.divider()
    st.subheader("Manage Accounts")
    new_account = st.text_input("New Account Name", key=f"{key_prefix}_new_account")
    add_account_button = st.button("Add Account", key=f"{key_prefix}_add_account")

    if add_account_button and new_account:
        if new_account not in st.session_state.accounts:
            st.session_state.accounts.append(new_account)
            save_accounts()
            st.rerun()

def main():
    st.title("AI Powered Personal Finance Coach")

    if 'df' not in st.session_state:
        st.session_state.df = None

    upload_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if upload_file is not None:
        st.session_state.df = load_transactions(upload_file)

    if st.session_state.df is not None:
        df = st.session_state.df

        if 'Transaction Type' in df.columns:
            debits_df = df[df['Transaction Type'] == 'debit'].copy()
            credits_df = df[df['Transaction Type'] == 'credit'].copy()

            tab1, tab2, tab3 = st.tabs(["Debit Transactions", "Credit Transactions", "Categories"])

            with tab1:
                st.header("Debit Transactions")
                st.dataframe(debits_df)
                st.divider()
                transaction_form('debit', df)
                manage_accounts_ui('debit')

            with tab2:
                st.header("Credit Transactions")
                st.dataframe(credits_df)
                st.divider()
                transaction_form('credit', df)
                manage_accounts_ui('credit')

            with tab3:
                st.header("Manage and View Categories")

                with st.expander("Add New Category"):
                    new_category = st.text_input("New Category Name")
                    add_button = st.button("Add Category")

                    if add_button and new_category:
                        if new_category not in st.session_state.categories:
                            st.session_state.categories[new_category] = []
                            save_categories()
                            st.rerun()
                
                st.subheader("View Transactions by Category")
                if 'Category' in df.columns:
                    category_options = df['Category'].dropna().unique().tolist()
                    selected_category = st.selectbox("Select a category", options=category_options)
                    
                    if selected_category:
                        category_df = df[df['Category'] == selected_category]
                        st.dataframe(category_df)
                else:
                    st.warning("No 'Category' column found in the dataframe.")

        else:
            st.warning("The CSV file must contain a 'Transaction Type' column with 'debit' and 'credit' values.")
            st.dataframe(df)

if __name__ == "__main__":
    main()
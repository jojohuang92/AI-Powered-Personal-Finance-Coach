import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="AI Powered Personal Finance Coach", page_icon="💰", layout="wide")

category_file = "categories.json"
account_file = "accounts.json"
budget_file = "budgets.json"

for file in [category_file, account_file, budget_file]:
    if os.path.exists(file):
        os.remove(file)

if "categories" not in st.session_state:
    if os.path.exists(category_file):
        with open(category_file, "r") as f:
            st.session_state.categories = json.load(f)
    else:
        st.session_state.categories = {}

if "accounts" not in st.session_state:
    if os.path.exists(account_file):
        with open(account_file, "r") as f:
            st.session_state.accounts = json.load(f)
    else:
        st.session_state.accounts = []

if "budgets" not in st.session_state:
    if os.path.exists(budget_file):
        with open(budget_file, "r") as f:
            st.session_state.budgets = json.load(f)
    else:
        st.session_state.budgets = {}

if "page" not in st.session_state:
    st.session_state.page = "main"

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def save_accounts():
    with open(account_file, "w") as f:
        json.dump(st.session_state.accounts, f)

def save_budgets():
    with open(budget_file, "w") as f:
        json.dump(st.session_state.budgets, f)

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

def main():
    st.title("AI Powered Personal Finance Coach")

    if 'df' not in st.session_state:
        st.session_state.df = None

    upload_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if upload_file is not None:
        st.session_state.df = load_transactions(upload_file)
        if st.session_state.df is not None and 'Category' in st.session_state.df.columns:
            csv_categories = st.session_state.df['Category'].dropna().unique().tolist()
            new_categories_added = False
            for category in csv_categories:
                if category not in st.session_state.categories:
                    st.session_state.categories[category] = []
                    new_categories_added = True
            if new_categories_added:
                save_categories()
                st.rerun()

        if st.session_state.df is not None and 'Account Name' in st.session_state.df.columns:
            csv_accounts = st.session_state.df['Account Name'].dropna().unique().tolist()
            new_accounts_added = False
            for account in csv_accounts:
                if account not in st.session_state.accounts:
                    st.session_state.accounts.append(account)
                    new_accounts_added = True
            if new_accounts_added:
                save_accounts()
                st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df

        if 'Transaction Type' in df.columns:
            debits_df = df[df['Transaction Type'] == 'debit'].copy()
            credits_df = df[df['Transaction Type'] == 'credit'].copy()

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Debit Transactions", "Credit Transactions", "Categories", "Budget", "Account"])

            with tab1:
                st.header("Debit Transactions")
                st.dataframe(debits_df)
                st.divider()
                transaction_form('debit', df)

            with tab2:
                st.header("Credit Transactions")
                st.dataframe(credits_df)
                st.divider()
                transaction_form('credit', df)

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
            
            with tab4:
                st.header("Set Your Budget")

                with st.form("budget_form"):
                    category_to_budget = st.selectbox("Category", options=list(st.session_state.categories.keys()))
                    budget_amount = st.number_input("Budget Amount", min_value=0.0, format="%.2f")
                    submit_budget = st.form_submit_button("Set Budget")

                    if submit_budget:
                        st.session_state.budgets[category_to_budget] = budget_amount
                        save_budgets()
                        st.success(f"Budget for {category_to_budget} set to ${budget_amount:.2f}")
                
                st.divider()
                st.subheader("Current Budgets")
                if st.session_state.budgets:
                    for category, budget in st.session_state.budgets.items():
                        st.write(f"{category}: ${budget:.2f}")
                else:
                    st.info("No budgets set yet.")
            
            with tab5:
                st.header("Manage Accounts")

                with st.expander("Add New Account"):
                    new_account = st.text_input("New Account Name", key="new_account_name_input")
                    add_account_button = st.button("Add Account")

                    if add_account_button and new_account:
                        if new_account not in st.session_state.accounts:
                            st.session_state.accounts.append(new_account)
                            save_accounts()
                            st.success(f"Account '{new_account}' added.")
                            st.rerun()
                        else:
                            st.warning(f"Account '{new_account}' already exists.")

                st.subheader("Your Accounts")
                df_accounts = []
                if 'Account Name' in df.columns:
                    df_accounts = df['Account Name'].dropna().unique().tolist()
                
                all_accounts = sorted(list(set(df_accounts + st.session_state.accounts)))

                if all_accounts:
                    for account in all_accounts:
                        st.write(account)
                else:
                    st.info("No accounts added yet. Accounts are also added automatically from your uploaded CSV.")


        else:
            st.warning("The CSV file must contain a 'Transaction Type' column with 'debit' and 'credit' values.")
            st.dataframe(df)

if __name__ == "__main__":
    main()
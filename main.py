import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

from chatbox import response
from forecasting import forecast
from nlp import extract_receipt
from anomaly_detection import anomaly

st.set_page_config(page_title="AI Powered Personal Finance Coach", page_icon="💰", layout="wide")

category_file = "categories.json"
account_file = "accounts.json"
budget_file = "budgets.json"
default_dataset_path = "dataset/personal_transactions.csv"

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
            if not description:
                st.error("Please enter a description")
                return
            if amount <= 0:
                st.error("Please enter a valid amount greater than 0")
                return
            if not category:
                st.error("Please select a category")
                return
            if not account_name:
                st.error("Please select an account")
                return

            try:
                transaction_id = f"{date}_{description}_{amount}_{transaction_type}_{account_name}"

                if 'last_added_transaction' not in st.session_state:
                    st.session_state.last_added_transaction = None

                if st.session_state.last_added_transaction == transaction_id:
                    return

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

                st.session_state.last_added_transaction = transaction_id

                st.success("Transaction added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding transaction: {str(e)}")

def main():
    st.title("AI Powered Personal Finance Coach")

    if 'df' not in st.session_state:
        if os.path.exists(default_dataset_path):
            st.session_state.df = load_transactions(default_dataset_path)
            if st.session_state.df is not None:
                if 'Category' in st.session_state.df.columns:
                    csv_categories = st.session_state.df['Category'].dropna().unique().tolist()
                    for category in csv_categories:
                        if category not in st.session_state.categories:
                            st.session_state.categories[category] = []
                    save_categories()

                if 'Account Name' in st.session_state.df.columns:
                    csv_accounts = st.session_state.df['Account Name'].dropna().unique().tolist()
                    for account in csv_accounts:
                        if account not in st.session_state.accounts:
                            st.session_state.accounts.append(account)
                    save_accounts()
        else:
            st.session_state.df = None

    upload_file = st.file_uploader("Upload your own transaction CSV file (Optional)", type=["csv"])

    if upload_file is not None:
        st.session_state.df = load_transactions(upload_file)
        needs_rerun = False

        if st.session_state.df is not None and 'Category' in st.session_state.df.columns:
            csv_categories = st.session_state.df['Category'].dropna().unique().tolist()
            for category in csv_categories:
                if category not in st.session_state.categories:
                    st.session_state.categories[category] = []
                    needs_rerun = True
            if needs_rerun:
                save_categories()

        if st.session_state.df is not None and 'Account Name' in st.session_state.df.columns:
            csv_accounts = st.session_state.df['Account Name'].dropna().unique().tolist()
            for account in csv_accounts:
                if account not in st.session_state.accounts:
                    st.session_state.accounts.append(account)
                    needs_rerun = True
            if needs_rerun:
                save_accounts()

        if needs_rerun:
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df

        if 'Transaction Type' in df.columns:
            debits_df = df[df['Transaction Type'] == 'debit'].copy()
            credits_df = df[df['Transaction Type'] == 'credit'].copy()

            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["Dashboard", "Debit Transactions", "Credit Transactions", "Categories", "Budget", "Account", "AI Insights", "Receipt Scanner"])

            with tab1:
                st.header("Financial Dashboard")

                total_income = credits_df[credits_df['Category'] == 'Paycheck']['Amount'].sum()

                total_expense = debits_df[debits_df['Category'] != 'Credit Card Payment']['Amount'].sum()
                net_savings = total_income - total_expense

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Income", f"${total_income:,.2f}")
                col2.metric("Total Expenses", f"${total_expense:,.2f}")
                col3.metric("Net Savings", f"${net_savings:,.2f}")

                #fix
                col4.metric("Average Daily Spending", f"${total_expense / 30:.2f}")

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Spending by Category")
                    if not debits_df.empty:
                        category_spending = debits_df.groupby('Category')['Amount'].sum().reset_index()
                        fig_cat_spending = px.pie(category_spending, 
                                                  values='Amount', 
                                                  names='Category', 
                                                  title='Spending Distribution Across Categories',
                                                  hole=.3)
                        st.plotly_chart(fig_cat_spending, width= "stretch")
                    else:
                        st.info("No debit transactions to display.")

                with col2:
                    st.subheader("Spending Over Time")
                    if not debits_df.empty:
                        debits_df_sorted = debits_df.sort_values('Date')
                        spending_over_time = debits_df_sorted.set_index('Date').resample('ME')['Amount'].sum()
                        st.line_chart(spending_over_time)
                    else:
                        st.info("No debit transactions to display.")

                st.divider()

                st.subheader("Spending Anomaly Detection")
                anomalous_spending = anomaly(debits_df)

                if not anomalous_spending.empty:
                    st.warning("We've detected some unusually high spending. Use the filter below to narrow down by category.")
                    
                    anomaly_categories = anomalous_spending['Category'].unique().tolist()
                    anomaly_categories.insert(0, "All Categories")
                    
                    selected_category = st.selectbox(
                        "Filter anomalies by category:",
                        options=anomaly_categories
                    )
                    
                    if selected_category == "All Categories":
                        st.dataframe(anomalous_spending[['Date', 'Description', 'Category', 'Amount']])
                    else:
                        st.dataframe(anomalous_spending[anomalous_spending['Category'] == selected_category][['Date', 'Description', 'Category', 'Amount']])
                else:
                    st.success("No spending anomalies detected. Great job staying on track!")


            with tab2:
                st.header("Debit Transactions")
                st.dataframe(debits_df)
                st.divider()
                transaction_form('debit', df)

            with tab3:
                st.header("Credit Transactions")
                st.dataframe(credits_df)
                st.divider()
                transaction_form('credit', df)

            with tab4:
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
            
            with tab5:
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
            
            with tab6:
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

                df_accounts = []
                if 'Account Name' in df.columns:
                    df_accounts = df['Account Name'].dropna().unique().tolist()
                
                all_accounts = sorted(list(set(df_accounts + st.session_state.accounts)))

                st.divider()

                st.subheader("View Spending by Account")
                if all_accounts:
                    selected_account_for_view = st.selectbox("Select an account to view transactions", options=all_accounts, key="view_account_select")

                    if selected_account_for_view:
                        account_df = df[df['Account Name'] == selected_account_for_view]
                        
                        if not account_df.empty:
                            account_debits = account_df[account_df['Transaction Type'] == 'debit']['Amount'].sum()
                            account_credits = account_df[account_df['Transaction Type'] == 'credit']['Amount'].sum()
                            
                            col1, col2 = st.columns(2)
                            col1.metric(f"Total Spending from {selected_account_for_view}", f"${account_debits:,.2f}")
                            col2.metric(f"Total Deposits to {selected_account_for_view}", f"${account_credits:,.2f}")

                            st.dataframe(account_df)

                            account_debits_df = account_df[account_df['Transaction Type'] == 'debit']
                            if not account_debits_df.empty:
                                st.subheader(f"Spending Categories for {selected_account_for_view}")
                                fig_account_spending = px.pie(account_debits_df, 
                                                              values='Amount', 
                                                              names='Category', 
                                                              title=f'Spending Breakdown for {selected_account_for_view}',
                                                              hole=.3)
                                st.plotly_chart(fig_account_spending, width= "stretch")
                        else:
                            st.info(f"No transactions found for account '{selected_account_for_view}'.")
                else:
                    st.info("No accounts available to view.")

            with tab7:
                st.header("AI-Powered Insights")

                st.subheader("Chat with your AI Financial Coach")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask a question about your finances..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            ai_response = response(st.session_state.df, prompt)
                            st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                st.divider()
                st.subheader("Financial Analysis")

                

            with tab8:
                st.header("Receipt Scanner")
                pass


        else:
            st.warning("The CSV file must contain a 'Transaction Type' column with 'debit' and 'credit' values.")
            st.dataframe(df)

if __name__ == "__main__":
    main()
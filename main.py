import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import pytesseract
from PIL import Image
from forecasting import frcst
import plotly.graph_objects as go


from chatbox import response, analysis
from forecasting import frcst
from forecasting import frcst, frcst_tot, frcstby_cat, _detect_trend, mt_frcst, _simple_average_forecast,get_budget_runway
from nlp import extract_receipt
from anomaly_detection import anomaly

st.set_page_config(page_title="AI Powered Personal Finance Coach", page_icon="💰", layout="wide")

category_file = "categories.json"
account_file = "accounts.json"
budget_file = "budgets.json"
default_dataset_path = "dataset/personal_transactions.csv"
default_budget_path = "dataset/Budget.csv"

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def save_accounts():
    with open(account_file, "w") as f:
        json.dump(st.session_state.accounts, f)

def save_budgets():
    with open(budget_file, "w") as f:
        json.dump(st.session_state.budgets, f)

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
    elif os.path.exists(default_budget_path):
        budget_df = pd.read_csv(default_budget_path)
        st.session_state.budgets = dict(zip(budget_df['Category'], budget_df['Budget']))
        save_budgets()
    else:
        st.session_state.budgets = {}

if "page" not in st.session_state:
    st.session_state.page = "main"

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

def transaction_form(transaction_type, df, defaults=None, form_key_suffix=""):
    if defaults is None:
        defaults = {}

    subheader_text = f"Add New {transaction_type.capitalize()} Transaction"
    if form_key_suffix: # If called from receipt scanner, use a different header
        subheader_text = "Add to Transactions"

    st.subheader(subheader_text)

    with st.form(f"new_{transaction_type}_transaction_form_{form_key_suffix}", clear_on_submit=True):
        date = st.date_input("Date", value=defaults.get("date"))
        description = st.text_input("Description", value=defaults.get("description"))
        
        category_options = list(st.session_state.categories.keys())
        category = st.selectbox("Category", options=category_options)
        
        amount = st.number_input("Amount", format="%.2f", min_value=0.0, value=defaults.get("amount", 0.0))

        df_accounts = []
        if 'Account Name' in df.columns:
            df_accounts = df['Account Name'].dropna().unique().tolist()
        
        all_accounts = sorted(list(set(df_accounts + st.session_state.accounts)))
        account_name = st.selectbox("Account Name", options=all_accounts)

        if "transaction_type" in defaults:
            type_options = ["debit", "credit"]
            type_index = type_options.index(defaults.get("transaction_type", "debit"))
            transaction_type = st.selectbox("Transaction Type", options=type_options, index=type_index)

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

    if st.session_state.get("df") is not None and "financial_analysis" not in st.session_state:
        df_copy = st.session_state.df.copy()
        st.session_state.financial_analysis = analysis(df_copy)


    upload_file = st.file_uploader("Upload your own transaction CSV file (Optional)", type=["csv"])

    if upload_file is not None:
        st.session_state.df = load_transactions(upload_file)
        if 'financial_analysis' in st.session_state:
            del st.session_state.financial_analysis

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

                debits_df = df[df['Transaction Type'] == 'debit'].copy()
                credits_df = df[df['Transaction Type'] == 'credit'].copy()

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



                st.divider()
                # wil work on the forecasitng secrion now .
                st.subheader("📊 Spending Forecast & Predictions")
    
                if not debits_df.empty:
                
                    forecast_data = frcst(df, frcst_m=3, trsnctn_ty='debit')
                    
                    # will be the metrix on the tp
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        next_month = forecast_data['total_forecast']['amounts'][0]
                        trend_emoji = "📈" if forecast_data['trend'] == 'increasing' else "📉" if forecast_data['trend'] == 'decreasing' else "➡️"
                        st.metric("Next Month Predicted", f"${next_month:,.2f}", delta=f"{forecast_data['trend']} {trend_emoji}")
                    
                    with col2:
                        avg_3m = forecast_data['total_forecast']['average']
                        st.metric("3-Month Average Forecast", f"${avg_3m:,.2f}")
                    
                    with col3:
                        hist_avg = forecast_data['past_avg']
                        diff = next_month - hist_avg
                        st.metric("vs Historical", f"${abs(diff):,.2f}", delta=f"{'Higher' if diff > 0 else 'Lower'}")
                    
                    # the chart given for the forecast
                    st.markdown("#### 📈 3-Month Spending Forecast")
                    
                    forecast_df = pd.DataFrame({
                        'Month': forecast_data['total_forecast']['dates'],
                        'Forecasted': forecast_data['total_forecast']['amounts'],
                        'Lower Bound': forecast_data['total_forecast']['lower_bound'],
                        'Upper Bound': forecast_data['total_forecast']['upper_bound']
                    })
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Month'], y=forecast_df['Upper Bound'],
                        fill=None, mode='lines', line_color='rgba(200,200,200,0.2)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Month'], y=forecast_df['Lower Bound'],
                        fill='tonexty', mode='lines', line_color='rgba(200,200,200,0.2)',
                        name='Confidence Interval', fillcolor='rgba(200,200,200,0.3)'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Month'], y=forecast_df['Forecasted'],
                        mode='lines+markers', name='Forecasted Spending',
                        line=dict(color='rgb(31, 119, 180)', width=3), marker=dict(size=10)
                    ))
                    
                    fig.update_layout(title='Predicted Spending', xaxis_title='Month', 
                                    yaxis_title='Amount ($)', hovermode='x unified', height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
                else:
                    st.info("Not enough data for forecasting. Add more transactions!")


            with tab2:
                st.header("Debit Transactions")
                debits_df = df[df['Transaction Type'] == 'debit'].copy()
                st.dataframe(debits_df)
                st.divider()
                transaction_form('debit', df)

            with tab3:
                st.header("Credit Transactions")
                credits_df = df[df['Transaction Type'] == 'credit'].copy()
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
                    col_cat, col_month = st.columns([2, 1])

                    with col_cat:
                        category_options = df['Category'].dropna().unique().tolist()
                        selected_category = st.selectbox("Select a category", options=category_options)

                    with col_month:
                        # Get all unique months from the dataset
                        df['YearMonth'] = df['Date'].dt.to_period('M')
                        all_months = sorted(df['YearMonth'].unique(), reverse=True)
                        month_options = ['All Months'] + [str(m) for m in all_months]
                        selected_month = st.selectbox("Select a month", options=month_options)

                    if selected_category:
                        category_df = df[df['Category'] == selected_category].copy()

                        # Filter by selected month if not "All Months"
                        if selected_month != 'All Months':
                            category_df = category_df[category_df['YearMonth'] == selected_month]

                        # Monthly spending analysis
                        st.divider()
                        if selected_month == 'All Months':
                            st.subheader(f"Monthly Spending for {selected_category}")
                        else:
                            st.subheader(f"Spending for {selected_category} in {selected_month}")

                        # Filter for debit transactions only
                        category_debits = category_df[category_df['Transaction Type'] == 'debit'].copy()

                        if not category_debits.empty:
                            if selected_month == 'All Months':
                                # Add Year-Month column
                                category_debits['YearMonth'] = category_debits['Date'].dt.to_period('M')

                                # Calculate monthly totals
                                monthly_spending = category_debits.groupby('YearMonth')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
                                monthly_spending.columns = ['Month', 'Total Spent', 'Transactions', 'Avg per Transaction']
                                monthly_spending['Month'] = monthly_spending['Month'].astype(str)

                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Spent", f"${category_debits['Amount'].sum():,.2f}")
                                col2.metric("Avg Monthly", f"${monthly_spending['Total Spent'].mean():,.2f}")
                                col3.metric("Total Transactions", f"{len(category_debits)}")
                                col4.metric("Avg per Transaction", f"${category_debits['Amount'].mean():,.2f}")

                                # Monthly spending table
                                st.markdown("#### Monthly Breakdown")
                                display_df = monthly_spending.copy()
                                display_df['Total Spent'] = display_df['Total Spent'].apply(lambda x: f"${x:,.2f}")
                                display_df['Avg per Transaction'] = display_df['Avg per Transaction'].apply(lambda x: f"${x:,.2f}")
                                st.dataframe(display_df, use_container_width=True, hide_index=True)

                                # Monthly spending chart
                                st.markdown("#### Spending Trend")
                                monthly_chart_data = category_debits.groupby('YearMonth')['Amount'].sum().reset_index()
                                monthly_chart_data['YearMonth'] = monthly_chart_data['YearMonth'].astype(str)
                                monthly_chart_data = monthly_chart_data.set_index('YearMonth')
                                st.line_chart(monthly_chart_data)
                            else:
                                # Single month view - show detailed metrics
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Spent", f"${category_debits['Amount'].sum():,.2f}")
                                col2.metric("Total Transactions", f"{len(category_debits)}")
                                col3.metric("Avg per Transaction", f"${category_debits['Amount'].mean():,.2f}")

                                # Show spending distribution for the month
                                st.markdown("#### Daily Spending")
                                daily_spending = category_debits.groupby(category_debits['Date'].dt.date)['Amount'].sum().reset_index()
                                daily_spending.columns = ['Date', 'Amount']
                                st.bar_chart(daily_spending.set_index('Date'))
                        else:
                            if selected_month == 'All Months':
                                st.info(f"No debit transactions found for {selected_category}")
                            else:
                                st.info(f"No debit transactions found for {selected_category} in {selected_month}")

                        # All transactions
                        st.divider()
                        if selected_month == 'All Months':
                            st.subheader("All Transactions")
                        else:
                            st.subheader(f"Transactions in {selected_month}")

                        if not category_df.empty:
                            st.dataframe(category_df)
                        else:
                            st.info(f"No transactions found for {selected_category} in {selected_month}")
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

                # Month selector for budget view
                col_header, col_month_select = st.columns([2, 1])
                with col_header:
                    st.subheader("Current Budgets")
                with col_month_select:
                    # Get all unique months from debits
                    if not debits_df.empty:
                        debits_df['YearMonth'] = debits_df['Date'].dt.to_period('M')
                        budget_months = sorted(debits_df['YearMonth'].unique(), reverse=True)
                        budget_month_options = [str(m) for m in budget_months]
                        # Default to most recent month
                        selected_budget_month = st.selectbox("View Period", options=budget_month_options, index=0, key="budget_month_select")
                    else:
                        selected_budget_month = None

                if st.session_state.budgets:
                    budget_df = pd.DataFrame(list(st.session_state.budgets.items()), columns=['Category', 'Budget'])

                    # Filter spending by selected month
                    if selected_budget_month and not debits_df.empty:
                        filtered_debits = debits_df[debits_df['YearMonth'] == selected_budget_month]
                    else:
                        filtered_debits = debits_df.copy()

                    spending_df = filtered_debits.groupby('Category')['Amount'].sum().reset_index().rename(columns={'Amount': 'Spent'})

                    budget_status_df = pd.merge(budget_df, spending_df, on='Category', how='left').fillna(0)
                    budget_status_df['Remaining'] = budget_status_df['Budget'] - budget_status_df['Spent']

                    # Summary metrics
                    total_budget = budget_status_df['Budget'].sum()
                    total_spent = budget_status_df['Spent'].sum()
                    total_remaining = total_budget - total_spent

                    st.markdown(f"### Summary for {selected_budget_month}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Budget", f"${total_budget:,.2f}")
                    col2.metric("Total Spent", f"${total_spent:,.2f}")
                    col3.metric("Total Remaining", f"${total_remaining:,.2f}")
                    col4.metric("Budget Usage", f"{(total_spent/total_budget*100 if total_budget > 0 else 0):.1f}%")

                    st.divider()

                    # Individual category budgets
                    for category, budget in st.session_state.budgets.items():
                        st.write(f"**{category}**")
                        category_row = budget_status_df[budget_status_df['Category'] == category]

                        if not category_row.empty:
                            spent = category_row['Spent'].iloc[0]
                            remaining = budget - spent
                        else:
                            spent = 0
                            remaining = budget

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Budget", f"${budget:,.2f}")
                        col2.metric("Spent", f"${spent:,.2f}")
                        col3.metric("Remaining", f"${remaining:,.2f}")

                        progress = min(spent / budget, 1.0) if budget > 0 else 0
                        st.progress(progress)

                        if spent > budget:
                            st.error(f"You are ${spent - budget:,.2f} over your budget for {category}!")
                        elif spent > budget * 0.9:
                            st.warning(f"You've used {(spent/budget*100):.1f}% of your budget for {category}")
                        else:
                            st.success(f"On track! {(spent/budget*100):.1f}% used")

                        st.divider()
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

                    if st.session_state.get('financial_analysis'):
                        st.subheader("Your Financial Analysis")
                        st.markdown(st.session_state.financial_analysis)
                    
                    st.divider()

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
                                ai_response = response(st.session_state.df, st.session_state.budgets, prompt)
                                st.markdown(ai_response)

                        st.session_state.messages.append({"role": "assistant", "content": ai_response})

            with tab8:
                st.header("Receipt Scanner")

                uploaded_file = st.file_uploader(
                    "Upload Receipt Image or Text File",
                    type=["png", "jpg", "jpeg", "txt"]
                )

                if uploaded_file is not None:
                    file_type = uploaded_file.type
                    st.write(f"File type detected: {file_type}")

                    if file_type == "text/plain":
                        try:
                            text = uploaded_file.read().decode("utf-8")
                        except Exception as e:
                            st.error(f"Error reading text file: {str(e)}")
                            text = None

                    elif "image" in file_type:
                        try:
                            image = Image.open(uploaded_file)
                            st.image(image, caption="Uploaded Receipt", width= 'stretch')
                            text = pytesseract.image_to_string(image)
                        except Exception as e:
                            st.error(f"Error reading image file: {str(e)}")
                            text = None
                    else:
                        st.error("Unsupported file type.")
                        text = None

                    if text:
                        with st.expander("View Raw Extracted Text"):
                            st.text(text)

                        try:
                            result = extract_receipt(text)
                            st.subheader("Extracted Receipt Details")
                            st.json(result)
                            
                            st.subheader("Transaction Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                if result.get("merchant"):
                                    st.metric("Merchant", result["merchant"])
                                if result.get("amount"):
                                    st.metric("Total Amount", f"${result['amount']}")
                            with col2:
                                if result.get("date"):
                                    st.metric("Date", result["date"])
                                if result.get("transaction_type"):
                                    st.metric("Type", result["transaction_type"].title())
                                    
                            st.divider()

                            # Prepare defaults for the transaction form
                            form_defaults = {
                                "date": pd.to_datetime(result.get("date")).date() if result.get("date") else None,
                                "description": result.get("merchant"),
                                "amount": float(result.get("amount", 0.0)),
                                "transaction_type": result.get("transaction_type", "debit")
                            }

                            # Call the reusable transaction_form function
                            # The transaction type from the form will be used, so the first argument is less critical here.
                            # We add a suffix to the form key to avoid conflicts with other forms.
                            transaction_form(form_defaults["transaction_type"], df, defaults=form_defaults, form_key_suffix="receipt")

                        except Exception as e:
                            st.error(f"Error running NLP extractor: {str(e)}")

        else:
            st.warning("The CSV file must contain a 'Transaction Type' column with 'debit' and 'credit' values.")
            st.dataframe(df)

if __name__ == "__main__":
    main()

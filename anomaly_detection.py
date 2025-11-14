import pandas as pd

def anomaly(debits_df: pd.DataFrame, min_threshold: float = 50.0, absolute_threshold: float = 500.0):
    if debits_df.empty or 'Category' not in debits_df.columns or 'Amount' not in debits_df.columns:
        return pd.DataFrame()

    iqr_anomalies_list = []
    
    for category, group in debits_df.groupby('Category'):
        if len(group) > 5:
            Q1 = group['Amount'].quantile(0.25)
            Q3 = group['Amount'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + (1.5 * IQR)
            
            category_anomalies = group[(group['Amount'] > upper_bound) & (group['Amount'] > min_threshold)]
            iqr_anomalies_list.append(category_anomalies)

    iqr_anomalies = pd.concat(iqr_anomalies_list) if iqr_anomalies_list else pd.DataFrame()

    absolute_anomalies = debits_df[debits_df['Amount'] > absolute_threshold]

    all_anomalies = pd.concat([iqr_anomalies, absolute_anomalies]).drop_duplicates().sort_values(by='Date', ascending=False)
    
    if all_anomalies.empty:
        return pd.DataFrame()
    
    return all_anomalies
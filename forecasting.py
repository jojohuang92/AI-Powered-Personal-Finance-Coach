import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

'''
We are making a future spending forecastbased on the data we gathered from the past .
basically getting more data ahead to have the coach work more efficiently with a
persnalized resul.t
'''

def frcst( df: pd.DataFrame, frcst_m: int = 3,trsnctn_ty: str= 'debit')-> Dict:
    if df.empty or 'Date' not in df.columns or 'Amount' not in df.columns:
        return mt_frcst(frcst_m)
    
    if 'Transaction Type' in df.columns:
        df = df[df['Transaction Type'] == trsnctn_ty].copy()

    if df.empty:
        return mt_frcst(frcst_m)
    
    df['Date']= pd.to_datetime(df['Date'])

    df['YearMonth']=  df['Date'].dt.to_period('M')
    m_tot = df.groupby('YearMonth')['Amount'].sum().sort_index()
    if len(m_tot) <2:
        return _simple_average_forecast(df, frcst_m)
    
    frcst_cat = frcstby_cat(df, frcst_m)
    tot_frcst = frcst_tot(m_tot, frcst_m)
    trnd = _detect_trend(m_tot)
    return {
        'forecast_category': frcst_cat,
        'total_forecast': tot_frcst,
        'forecast_months': frcst_m,
        'trend': trnd,
        'transaction_type': trsnctn_ty,
        'past_avg': m_tot.mean()
    }

# we're gettin the spending forecast for every single category used here
def frcstby_cat(df: pd.DataFrame, frcst_m: int)-> Dict:
    if 'Category' not in df.columns:
        return {}
    frcst_cat= {}
    df['YearMonth']=  df['Date'].dt.to_period('M')
    for category in df['Category'].unique():
        if pd.isna(category):
            continue
        cat_df = df[df['Category']==category]
        m_cat = cat_df.groupby('YearMonth')['Amount'].sum().sort_index()
        if len(m_cat)<2:
            avg = cat_df['Amount'].mean()
            frcst_val = [avg]*frcst_m
        else:
            w = np.exp(np.linspace(-1, 0, len(m_cat)))
            w/= w.sum()
            w_avg = np.sum(m_cat.values*w)
            x = np.arange(len(m_cat))
            y =m_cat.values
            z= np.polyfit(x, y, 1)
            trnd_sl = z[0]

            frcst_val= []
            for i in range(1, frcst_m+1):
                val =w_avg+(trnd_sl*i)
                val =max(0, val)
                frcst_val.append(val)

        frcst_cat[category]={
            'forecasted_amounts':frcst_val,
            'historical_average': m_cat.mean() if len(m_cat)>0 else cat_df['Amount'].mean(),
            'last_month': m_cat.iloc[-1] if len(m_cat)>0 else 0          
        } 
    return frcst_cat

def frcst_tot(m_tot: pd.Series, frcst_m:int)->Dict:
    v= m_tot.values
    alpha=0.3
    smoothed =[v[0]]
    for i in range(1, len(v)):
        smoothed.append(alpha*v[i]+(1-alpha)*smoothed[i-1])

    x = np.arange(len(v))
    z = np.polyfit(x, v, 1)
    trns_sl = z[0]
    trnd_int = z[1]
    last_x = len(v) - 1
    frcst_value = []
    l_bnd = []
    u_bnd = []
    std = np.std(v)

    for i in range(1, frcst_m + 1):
        trnd_frcst = trnd_int + trns_sl * (last_x + i)
        frcst_val = 0.7 * trnd_frcst + 0.3 * smoothed[-1]
        frcst_val = max(0, frcst_val)
        frcst_value.append(frcst_val)
        l_bnd.append(max(0, frcst_val - 1.96 * std))
        u_bnd.append(frcst_val + 1.96 * std)
    
    # Generate forecast dates
    last_date = m_tot.index[-1].to_timestamp()
    forecast_dates = [(last_date + pd.DateOffset(months=i)).strftime('%Y-%m') 
                     for i in range(1, frcst_m + 1)]
    
    return {
        'amounts': frcst_value,
        'dates': forecast_dates,
        'lower_bound': l_bnd,
        'upper_bound': u_bnd,
        'average': np.mean(frcst_value)
    }

# just normal detection of the trend to make sure and chec the increas, decrease or any type
# of change in the trend..
def _detect_trend(monthly_totals: pd.Series) -> str:
    """Detect if spending is increasing, decreasing, or stable."""
    
    if len(monthly_totals) < 3:
        return 'stable'
    
    # using on the past 6 months or less if not that  much data is available
    recent_months = min(6, len(monthly_totals))
    recent_data = monthly_totals.iloc[-recent_months:]
    
    # Linear regression
    x = np.arange(len(recent_data))
    y = recent_data.values
    z = np.polyfit(x, y, 1)
    trnd_sl= z[0]
    
    # we can just get the trend usingslope and significance
    avg = recent_data.mean()
    threshold = avg * 0.05  # 5% of average
    
    if trnd_sl > threshold:
        return 'increasing'
    elif trnd_sl < -threshold:
        return 'decreasing'
    else:
        return 'stable'


def mt_frcst(forecast_months: int) -> Dict:
    """Return empty forecast structure when no data is available."""
    
    return {
        'category_forecasts': {},
        'total_forecast': {
            'amounts': [0] * forecast_months,
            'dates': [],
            'lower_bound': [0] * forecast_months,
            'upper_bound': [0] * forecast_months,
            'average': 0
        },
        'forecast_months': forecast_months,
        'trend': 'stable',
        'historical_average': 0,
        'transaction_type': 'debit'
    }


def _simple_average_forecast(df: pd.DataFrame, forecast_months: int) -> Dict:
    """Simple average-based forecast when insufficient historical data."""
    
    avg_monthly = df['Amount'].sum() / max(1, len(df['Date'].dt.to_period('M').unique()))
    
    return {
        'category_forecasts': {},
        'total_forecast': {
            'amounts': [avg_monthly] * forecast_months,
            'dates': [],
            'lower_bound': [avg_monthly * 0.8] * forecast_months,
            'upper_bound': [avg_monthly * 1.2] * forecast_months,
            'average': avg_monthly
        },
        'forecast_months': forecast_months,
        'trend': 'stable',
        'historical_average': avg_monthly,
        'transaction_type': 'debit'
    }


def get_budget_runway(
    df: pd.DataFrame, 
    budgets: Dict[str, float],
    bal_cur: Optional[float] = None
) -> Dict:
    """
    Calculate how long current savings will last based on spending patterns.
    
    Args:
        df: Transaction DataFrame
        budgets: Dictionary of categorybudgets
        current_balance: Current account balanc
    
    Returns:
        Dictionary with runway estimates
    """
    
    if df.empty or not budgets:
        return {'runway_months':0, 'status' : 'insufficient_data'}
    
    #Get recent monthly spendng (last 3 month
    df = df[df['Transaction Type'] == 'debit'].copy()
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
    past_m = df['YearMonth'].unique()[-3:] if len(df['YearMonth'].unique()) >= 3 else df['YearMonth'].unique()
    
    recent_df = df[df['YearMonth'].isin(past_m)]
    avg_m_spndg = recent_df.groupby('YearMonth')['Amount'].sum().mean()
    
    if bal_cur and bal_cur > 0 and avg_m_spndg > 0:
        runway_m = bal_cur / avg_m_spndg
        return {
            'runway_months': round(runway_m, 1),
            'avg_monthly_spending': round(avg_m_spndg, 2),
            'current_balance': bal_cur,
            'status': 'ok' if runway_m > 3 else 'warning'
        }
    
    return {
        'runway_months': 0,
        'avg_monthly_spending': round(avg_m_spndg, 2) if avg_m_spndg else 0,
        'status': 'no_balance_provided'
    }



#!/usr/bin/env python
# coding: utf-8

# # Uploading data from API

# In[1]:


import pandas as pd
import os
import numpy as np
import math
import eventstudy as es
import refinitiv.dataplatform as rdp
import matplotlib.pyplot as plt
from refinitiv.dataplatform import historical_pricing
from refinitiv.dataplatform import Intervals
import refinitiv.dataplatform.eikon as ek
print(es.__version__)
from eventstudy import excelExporter
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime as dt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# In[2]:


rdp.open_desktop_session('API KEY HERE')


# In[3]:


#Retrieving Data from Refinitiv API
start_date = '2021-01-01'
end_date = '2023-05-23'

df_workspace, err = ek.get_data(
    instruments=['LCOc1', 'CLc1', 'HOc1', 'Wc1', 'Sc1', 'RBc1', 'CMZNc1', 'TRGBNBPMc1', 'KCc1', 'CMCUc1', 'SPY'],
    fields=['TR.CLOSEPRICE', 'TR.CLOSEPRICE.date'],
    parameters={'SDate': start_date, 'EDate': end_date, 'Curn': 'USD'}
)


# In[4]:


# Converting Date to a desired format to avoid inconsistencies when matching the data
df_workspace['Date'] = pd.to_datetime(df_workspace['Date']).dt.date

# Using PIVOT to match the format of the data according to the Excel data
df_api = df_workspace.pivot_table(index='Date', columns='Instrument', values='Close Price')

# Reorder the columns to match the desired order
columns_order = ['LCOc1', 'CLc1', 'HOc1', 'Wc1', 'Sc1', 'RBc1', 'CMZNc1', 'TRGBNBPMc1', 'KCc1', 'CMCUc1', 'SPY']
df_api = df_api[columns_order]

# Fill NaN values with the previous price
df_api = df_api.ffill()

# Reset the index for the date
df_api = df_api.reset_index()
df_api


# # Calculate Returns

# In[21]:


# Convert columns to numeric data type and handle missing values
numerical_columns = ['LCOc1', 'CLc1', 'HOc1', 'Wc1', 'Sc1', 'RBc1', 'CMZNc1', 'TRGBNBPMc1', 'KCc1', 'CMCUc1', 'SPY']
df_api[numerical_columns] = df_api[numerical_columns].fillna(0).astype(float)

# Calculate returns for all columns
df_returns = df_api[numerical_columns].pct_change()

# Include the date column in the returns DataFrame
df_returns.insert(0, 'Date', df_api['Date'])

# Reorder the columns to have 'Date' as the first column
df_returns = df_returns.reindex(columns=['Date'] + numerical_columns)
df_returns = df_returns.replace([np.inf, -np.inf], np.nan)
df_returns = df_returns.ffill()
df_returns.dropna(inplace=True)


# # Running the Event Study

# In[18]:


event_dates = pd.read_csv('FILE PATH + FOMC dates-github.csv')

event_date['event_date'] = pd.to_datetime(event_date['event_date']).dt.date
print(event_date)


# In[7]:


#Log returns calculation
tickers = ['LCOc1', 'CLc1', 'HOc1', 'Wc1', 'Sc1', 'RBc1', 'CMZNc1', 'TRGBNBPMc1', 'KCc1', 'CMCUc1', 'SPY']

df_log_returns = df_returns.copy() 
df_log_returns[tickers] = np.log(1 + df_log_returns[tickers])
df_log_returns.rename(columns={ticker: 'log_' + ticker for ticker in tickers}, inplace=True)

# Replace 'inf' values with NaN
df_log_returns = df_log_returns.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values
df_log_returns.dropna(inplace=True)


# In[23]:


#No 1 Event study using 10 days event window using an estimation window of 60 days
from scipy import stats
def compute_market_model_parameters(df, ticker, market_ticker):
    # Perform linear regression
    beta, alpha = np.polyfit(df[market_ticker], df[ticker], 1)
    return alpha, beta

def perform_t_test(abnormal_returns):
    # Perform a t-test for the null hypothesis that the mean is zero
    t_stat, p_value = stats.ttest_1samp(abnormal_returns, 0)
    return t_stat, p_value


event_window = list(range(-10, 11))  # From 10 days before the event to 10 days after

results = {'day': [], 'aar': [], 'caar': [], 't_stat': [], 'p_value': []}

# Iterate over each day in the event window
for day in event_window:
    # Initialize a list to store the abnormal returns for the current day
    abnormal_returns = []

    # Iterate over each row in the event_dates DataFrame
    for index, row in event_dates.iterrows():
        event_date = pd.to_datetime(row['event_date'])
        ticker = row['Ticker']
        market_ticker = row['Market']

        # Define the start and end dates of the estimation window (e.g., 60 days before the event)
        est_start_date = event_date - pd.DateOffset(days=60)
        est_end_date = event_date - pd.DateOffset(days=1)

        # Fetch the return data for the ticker and the market during the estimation window
        est_window_returns = df_returns[(df_returns['Date'] >= est_start_date) & (df_returns['Date'] <= est_end_date)][['Date', ticker, market_ticker]]

        # Compute the market model parameters (alpha and beta) using a linear regression
        alpha, beta = compute_market_model_parameters(est_window_returns, ticker, market_ticker)

        # Define the date of the current day in the event window
        current_date = event_date + pd.DateOffset(days=day)

        # Fetch the return data for the ticker and the market for the current day
        current_day_returns = df_returns[df_returns['Date'] == current_date][['Date', ticker, market_ticker]]

        # Computing the abnormal return for the current day
        if not current_day_returns.empty:
            # Compute the expected return based on the market model
            expected_return = alpha + beta * current_day_returns[market_ticker]
            actual_return = current_day_returns[ticker]
            abnormal_return = actual_return - expected_return
            abnormal_returns.append(abnormal_return)

    # Computing the AAR for the current day
    aar = np.mean(abnormal_returns)

    # Computing the CAAR up to the current day
    caar = np.sum(results['aar'] + [aar])

    # Performing the t-test
    t_stat, p_value = perform_t_test(abnormal_returns)

    # Append the results for the current day to the results dictionary
    results['day'].append(day)
    results['aar'].append(aar)
    results['caar'].append(caar)
    results['t_stat'].append(t_stat)
    results['p_value'].append(p_value)


results_df = pd.DataFrame(results)


# In[24]:


#Calculating the  90% confidence interval of AAR and CAAR
def calculate_confidence_interval(data, confidence=0.90):
    """Calculate the confidence interval for a given dataset."""
    mean = np.mean(data)
    standard_error = np.std(data) / np.sqrt(len(data))
    z_value = 1.645  # Z-value for a 90% confidence interval

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - (z_value * standard_error)
    upper_bound = mean + (z_value * standard_error)

    return lower_bound, upper_bound, standard_error

results = {'day': [], 'aar': [], 'aar_lower': [], 'aar_upper': [], 'std_err_aar': [], 'caar': [], 'caar_lower': [], 'caar_upper': [], 'std_err_caar': [], 't_stat': [], 'p_value': []}

# Iterate over each day in the event window
for day in event_window:
    # Compute the AAR for the current day
    aar = np.mean(abnormal_returns)

    # Compute the CAAR up to the current day
    caar = np.sum(results['aar'] + [aar])

    # Compute the confidence interval for AAR and its standard error
    aar_lower_bound, aar_upper_bound, std_err_aar = calculate_confidence_interval(abnormal_returns, confidence=0.90)

    # Compute the confidence interval for CAAR and its standard error
    caar_lower_bound, caar_upper_bound, std_err_caar = calculate_confidence_interval(results['caar'] + [caar], confidence=0.90)

    # Append the results for the current day to the results dictionary
    results['day'].append(day)
    results['aar'].append(aar)
    results['aar_lower'].append(aar_lower_bound)
    results['aar_upper'].append(aar_upper_bound)
    results['std_err_aar'].append(std_err_aar)
    results['caar'].append(caar)
    results['caar_lower'].append(caar_lower_bound)
    results['caar_upper'].append(caar_upper_bound)
    results['std_err_caar'].append(std_err_caar)
    results['t_stat'].append(t_stat)
    results['p_value'].append(p_value)

results_df = pd.DataFrame(results)


# In[ ]:





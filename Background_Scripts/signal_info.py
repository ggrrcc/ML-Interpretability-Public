# DO NOT REMOVE
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def flatten(l):
    f = []
    for m in l: 
        if len(m)>1: # If m is a list (i.e., contains multiple elements)
            for n in m:
                f.append(n)
        else:
            f.extend(m) # Otherwise, just extend the list f with the elements of m
    return f


def create_signal_label_dict(merged_data):
    # -------------------------------------------------------------------------------------------------
    # Signal Label Dictionary Initialization
    # -------------------------------------------------------------------------------------------------
    # Signals- The key is the buy/hold/sell label column name, and the value(s) is/are the numeric column name(s)
    # Format: signal_label_dict_og['YOUR_SIGNAL_HERE_Label'] = ['YOUR_SIGNAL_HERE']
    # Format if multiple: signal_label_dict_og['YOUR_SIGNAL_HERE_Label'] = ['Input_#_1', 'Input_#_2', 'etc']
    signal_label_dict_og = {}
    signal_label_dict_og['Altman_ZScore_Label'] = ['Altman_ZScore']
    signal_label_dict_og['RE_over_TD_Label'] = ['RE_over_TD']
    signal_label_dict_og['EPS_Old_Label'] = ['EPS_Old']
    signal_label_dict_og['Momentum_Label'] = ['Value', 'Growth', 'Quality', 'K_Score']
    signal_label_dict_og['Moving_Average_Label'] = ['Moving_Average']
    signal_label_dict_og['Price_Acceleration_Label'] = ['Price_Acceleration']

    # Addison
    signal_label_dict_og['Net_Income_Label'] = ['Net_Income']
    # signal_label_dict_og['State_Province_Label'] = ['State_Province']
    signal_label_dict_og['Operating_Activities_CF_Label'] = ['Operating_Activities_CF']
    signal_label_dict_og['Gross_Margin_Label'] = ['Gross_Margin']
    signal_label_dict_og['Debt_To_Equity2_Label'] = ['Debt_To_Equity2']

    # Antony
    signal_label_dict_og['Return_On_Equity_Label'] = ['Return_On_Equity']
    signal_label_dict_og['Return_On_Assets_Label'] = ['Return_On_Assets']
    signal_label_dict_og['Operating_Efficiency_Label'] = ['Operating_Efficiency']
    signal_label_dict_og['Leverage_Ratio_Label'] = ['Leverage_Ratio']
    signal_label_dict_og['Interest_Coverage_Ratio_Label'] = ['Interest_Coverage_Ratio']

    # Ben
    signal_label_dict_og['Total_Shares_Repurchased_Label'] = ['Total_Shares_Repurchased']
    # signal_label_dict_og['In_Process_R_D_Label'] = ['In_Process_R_D']
    signal_label_dict_og['Capital_Expenditures_Label'] = ['Capital_Expenditures']
    signal_label_dict_og['Cash_Dividends_Label'] = ['Cash_Dividends']
    signal_label_dict_og['Deferred_Revenue_Label'] = ['Deferred_Revenue']

    # Jackson
    signal_label_dict_og['Debt_Label'] = ['Debt']
    signal_label_dict_og['Debt_To_Equity_Ratio_Label'] = ['Debt_To_Equity_Ratio']
    signal_label_dict_og['EBITDA_Label'] = ['EBITDA']
    signal_label_dict_og['Enterprise_Value_Label'] = ['Enterprise_Value']
    signal_label_dict_og['Enterprise_Multiple_Label'] = ['Enterprise_Multiple']

    # Rohan
    signal_label_dict_og['Volume_Label'] = ['Volume']
    signal_label_dict_og['EPS_New_Label'] = ['EPS_New']
    signal_label_dict_og['PE_Ratio_Label'] = ['PE_Ratio']
    signal_label_dict_og['RSI_Label'] = ['RSI']
    signal_label_dict_og['Bollinger_Bands_Label'] = ['Upper_Band', 'Lower_Band']
    # signal_label_dict_og['Bollinger_Bands_Label'] = ['Upper_Band', 'Lower_Band', 'Rolling_Mean', 'Rolling_Std'] # ALTERNATIVE: Can even include current price?

    # Ryan
    signal_label_dict_og['Current_Assets_Label'] = ['Current_Assets']
    signal_label_dict_og['Cash_Flow_Model_Label'] = ['Cash_Flow_Model']
    signal_label_dict_og['Income_Taxes_Label'] = ['Income_Taxes']
    signal_label_dict_og['Acquisitions_Label'] = ['Acquisitions']
    signal_label_dict_og['Depreciation_Amortization_Label'] = ['Depreciation_Amortization']
    # signal_label_dict_og["Funds_From_Operations_Label"] = ["Funds_From_Operations"]
    
    # Ved
    signal_label_dict_og['Profit_Margin_Label'] = ['Profit_Margin']
    signal_label_dict_og['Risk_Adjusted_Return_Label'] = ['Risk_Adjusted_Return', 'Risk_Free_Rate']
    signal_label_dict_og['Total_Invested_Capital_Label'] = ['Total_Invested_Capital']
    signal_label_dict_og['Net_Receivables_Label'] = ['Net_Receivables']
    signal_label_dict_og['ROIC_Label'] = ['ROIC']

    ########################################################################################################################
    ########################################## Any changes to be added above this ##########################################
    ########################################################################################################################
    signal_label_dict_og['Gen_Label'] = flatten(list(signal_label_dict_og.values()))

    # -------------------------------------------------------------------------------------------------
    # Signals and Strategy Configuration
    # -------------------------------------------------------------------------------------------------
    if not('current_signals' in locals() or 'current_signals' in globals()):
        current_signals = signal_label_dict_og.keys()
        num_signals = len(list(signal_label_dict_og.keys()))
    else:
        current_signals = signal_label_dict.keys()
        num_signals = len(list(signal_label_dict.keys()))
        
    ### Fitting original models- the 'all' column is used for creating models that use all the data (either stocks or shares)
    stocks = sorted(merged_data['permno'].unique())
    num_stocks = len(stocks)
    min_accounting_lag = 1
    # Info about the strategy, used for ex-post statistics and output not the actual backtest
    strategy_info = {
        'brief descriptor': '{0}_stock_{1}signal'.format(num_stocks,num_signals), 
        'plot descriptor': 'B/H/S Strategy, Equal-Weighted',
        'universe': 'Public US equities with accounting data',
        'signals': current_signals, #, measured at most recent earnings announcement??
        'trading rule': 'Buy "A" stocks, sell "F" stocks', # What kind of weight? equal, rank, value?
        'holding period': 'One month',
        'periods per year': 12,
        'time lag': 'Minimum of {0} days from announcement of quarterly earnings'.format(min_accounting_lag),
        'output folder name': 'Outputs'
    }

    return signal_label_dict_og, strategy_info

def add_signals_to_df(merged_df, small_ret, big_ret, default_buy_threshold, default_sell_threshold):
    import numpy as np
    import pandas as pd

    signal_label_dict_og, _ = create_signal_label_dict(merged_df)
    general_columns_list = ['permno', 'PRC', 'date', 'datadate', 'rdq', 'Future_Close', 'Future_Price_Change', 'Gen_Label', 'RET', 'naics']
    # Labeling for buy, hold, sell based on future price movement
    look_forward_days = small_ret
    merged_df['Future_Close'] = merged_df['PRC'].shift(-look_forward_days)
    merged_df['Future_Price_Change'] = (merged_df['Future_Close'] - merged_df['PRC']) / merged_df['PRC']

    # General signal
    merged_df['Gen_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Gen_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Gen_Label'] = 'sell'

    signal_list = list(set(list(signal_label_dict_og.keys()))-set(['Gen_Label']))
    signal_label_list = signal_label_dict_og['Gen_Label']
    
    '''
    ############################
    # Calculate Signal Example #
    ############################
    # Calculate your signal
    merged_df['Signal'] = Whatever you do here to calculate your signal. Add, divide, shift, find difference, etc

    # Optional: set a new buy_threshold and sell threshold. There's a default one (default_buy_threshold and default_sell_threshold)
    # Optional: Define B/H/S Threshold
    buy_threshold_signal = 0.015   # adjustable
    sell_threshold_signal = -0.015   # adjustable

    merged_df['Signal_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Signal_Label'] = 'buy' ### or buy_threshold_signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Signal_Label'] = 'sell' ### or sell_threshold_signal
    '''
    #####################################################################################################################################################################################################################################################################
    '''''''''
    Add your signals below
    '''''''''
    
    '''
    Previously done
    '''
    ####################
    # Calculate Altman #
    ####################
    calc_cols = ['wcapq', 'atq', 'req', 'oiadpq', 'revtq', 'ltq', 'PRC', 'SHROUT']
    # Get rid of rows that are missing too much data
    # We probably need to do this for most models
    merged_df['numNull'] = merged_df[calc_cols].isna().sum(1)
    merged_df = merged_df[merged_df['numNull']<=2]
    # Impute missing data
    for i in calc_cols:
        mean = merged_df[i].mean()
        merged_df[i].fillna(value = mean, inplace = True)

    merged_df['Altman_ZScore'] = 1.2*merged_df['wcapq']/merged_df['atq'] + 1.4*merged_df['req']/merged_df['atq']+3.3*merged_df['oiadpq']/merged_df['atq']+merged_df['revtq']/merged_df['atq']+1.6*merged_df['PRC']*merged_df['SHROUT']/merged_df['atq']/1000

    # # Define B/H/S Threshold
    # buy_threshold_azs = 0.015   # adjustable
    # sell_threshol_azs = -0.015   # adjustable
    merged_df['Altman_ZScore_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Altman_ZScore_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Altman_ZScore_Label'] = 'sell'

    ########################
    # Calculate RE_over_TD # # Todo
    ########################
    # Need the +1 just in case there's no debt to avoid this being infinity
    merged_df['RE_over_TD'] = merged_df['req']/(merged_df['ltq']+1)

    merged_df['RE_over_TD_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'RE_over_TD_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'RE_over_TD_Label'] = 'sell'

    #################
    # Calculate EPS #
    #################
    merged_df['EPS_Old'] = merged_df['niq'] / (merged_df['SHROUT']*1000)

    # Define B/H/S Threshold
    buy_threshold_eps = 0.015   # adjustable
    sell_threshold_eps = -0.015   # adjustable
    merged_df['EPS_Old_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > buy_threshold_eps, 'EPS_Old_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < sell_threshold_eps, 'EPS_Old_Label'] = 'sell'

    ######################
    # Calculate Momentum #
    ######################
    # Calculate indicators as proxies for K_Score components
    merged_df['Momentum'] = merged_df['PRC'].pct_change(small_ret)  # 1 month returns for momentum
    merged_df['Value'] = merged_df['PRC'] / merged_df['PRC'].rolling(window=big_ret).mean()  # Price to yearly mean as value proxy
    merged_df['Growth'] = merged_df['PRC'].pct_change(big_ret)  # Year-over-year growth
    merged_df['Quality'] = merged_df['VOL'].diff() / (merged_df['VOL']+1)  # Change in volume as quality proxy
    # Simplified K_Score as an average of normalized factors (for illustration only)
    merged_df['K_Score'] = (merged_df[['Momentum', 'Value', 'Growth', 'Quality']].apply(lambda x: (x - x.mean()) / x.std()).mean(axis=1))

    merged_df['Momentum_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Momentum_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Momentum_Label'] = 'sell'

    ################
    # Calculate MA #
    ################
    window_size = 20
    merged_df['Moving_Average'] = merged_df['PRC'].rolling(window=window_size).mean()

    # Define threshold for moving average signal
    threshold_ma = 0.015  # Adjust threshold value as needed
    # Define buy and sell conditions based on the relationship between closing price and moving average
    merged_df['Moving_Average_Label'] = 'hold'
    merged_df.loc[merged_df['PRC'] > (1 + threshold_ma) * merged_df['Moving_Average'], 'Moving_Average_Label'] = 'buy'
    merged_df.loc[merged_df['PRC'] < (1 - threshold_ma) * merged_df['Moving_Average'], 'Moving_Average_Label'] = 'sell'

    ################################
    # Calculate Price Acceleration #
    ################################
    merged_df['Price_Acceleration'] = merged_df['PRC'].diff(small_ret) / merged_df['PRC'].shift(small_ret)

    # Define buy and sell conditions based on price acceleration
    merged_df['Price_Acceleration_Label'] = 'hold'
    merged_df.loc[merged_df['Price_Acceleration'] > default_buy_threshold, 'Price_Acceleration_Label'] = 'buy'
    merged_df.loc[merged_df['Price_Acceleration'] < default_sell_threshold, 'Price_Acceleration_Label'] = 'sell'

    '''
    Addison
    '''
    ############################
    # Calculate Net Income Signal #
    ############################
    merged_df['Net_Income'] = merged_df['niq']
    # Optional: set a new buy_threshold and sell threshold. There's a default one (default_buy_threshold and default_sell_threshold)
    # Optional: Define B/H/S Threshold
    # buy_threshold_signal = 0.015   # adjustable
    # sell_threshold_signal = -0.015   # adjustable

    merged_df['Net_Income_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Net_Income_Label'] = 'buy' ### or buy_threshold_signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Net_Income_Label'] = 'sell' ### or sell_threshold_signal

    ############################
    # State / Province Location #
    ############################
  
    # label encoder = label encoder
    # merge df encoded state = label encoder (merged_df state_province)
    
    label_encoder = LabelEncoder()
    merged_df['State_Province'] = label_encoder.fit_transform(merged_df['state'])

    merged_df['State_Province_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'State_Province_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'State_Province_Label'] = 'sell'

    merged_df['State_Province_Label_Encoded'] = label_encoder.fit_transform(merged_df['State_Province_Label'])

    ############################
    # Operating Activities Net CF #
    ############################
    merged_df['Operating_Activities_CF'] = merged_df['oancfy']
   

    merged_df['Operating_Activities_CF_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Operating_Activities_CF_Label'] = 'buy' ### or buy_threshold_signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Operating_Activities_CF_Label'] = 'sell' ### or sell_threshold_signal

    ############################
    # Gross Margin Yearly #
    ############################
    merged_df['Gross_Margin'] = (merged_df['revty'] - merged_df['cogsy']) / merged_df['revty']
   

    merged_df['Gross_Margin_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Gross_Margin_Label'] = 'buy' ### or buy_threshold_signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Gross_Margin_Label'] = 'sell' ### or sell_threshold_signal
    
    ############################
    # Debt to Equity #
    ############################
    merged_df['Debt_To_Equity2'] = merged_df['ltq'] / merged_df['teqq']
   

    merged_df['Debt_To_Equity2_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Debt_To_Equity2_Label'] = 'buy' ### or buy_threshold_signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Debt_To_Equity2_Label'] = 'sell' ### or sell_threshold_signal


    '''
    Antony
    '''

    ##############################
    # Calculate Return on Equity #
    ##############################
    # ROE measures how efficiently a company generates profit from shareholders' equity.  
    # It was chosen because a consistently high ROE indicates strong profitability and effective capital use.  
    merged_df['Return_On_Equity'] = merged_df['niq']/ (merged_df['teqq'])

    # Define buy and sell conditions based on Return on Equity  
    merged_df['Return_On_Equity_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Return_On_Equity_Label'] = 'buy'  
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Return_On_Equity_Label'] = 'sell'  

    ####################################
    # Calculate Return on Assets (ROA) #
    ####################################
    # ROA shows how effectively a company turns its assets into profit.  
    # It was chosen because a higher ROA suggests strong operational efficiency, making it a key indicator of financial health.  
    merged_df['Return_On_Assets'] = merged_df['niq']/(merged_df['atq'])

    # Define buy and sell conditions based on Return on Assets  
    merged_df['Return_On_Assets_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Return_On_Assets_Label'] = 'buy'  
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Return_On_Assets_Label'] = 'sell'  

    ##################################
    # Calculate Operating Efficiency #
    ##################################
    # A higher operating efficiency means a company is using its revenue more effectively to generate profit.  
    # It was chosen because efficient companies tend to have higher margins and are better equipped to handle downturns.  
    merged_df['Operating_Efficiency'] = merged_df['oibdpy']/(merged_df['revty']+1)

    # Define buy and sell conditions based on Operating Efficiency  
    merged_df['Operating_Efficiency_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Operating_Efficiency_Label'] = 'buy'  
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Operating_Efficiency_Label'] = 'sell'  

    ############################
    # Calculate Leverage Ratio #
    ############################
    # The leverage ratio indicates financial risk by measuring how much debt a company uses relative to its assets.  
    # It was chosen because excessive leverage increases financial risk, while lower leverage suggests stability.  
    merged_df['Leverage_Ratio'] = (merged_df['dlcchy']+merged_df['dltisy'])/(merged_df['atq'])

    # Define buy and sell conditions based on Leverage Ratio  
    merged_df['Leverage_Ratio_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] < default_buy_threshold, 'Leverage_Ratio_Label'] = 'buy'  
    merged_df.loc[merged_df['Future_Price_Change'] > default_sell_threshold, 'Leverage_Ratio_Label'] = 'sell'  

    #####################################
    # Calculate Interest Coverage Ratio #
    #####################################
    # A higher Interest Coverage Ratio means a company is in a better position to pay its debt interest.  
    # It was chosen because companies with strong interest coverage are less likely to face liquidity issues or default.  
    merged_df['Interest_Coverage_Ratio'] = merged_df['oiadpy']/(merged_df['xintq']+1)

    # Define buy and sell conditions based on Interest Coverage Ratio  
    merged_df['Interest_Coverage_Ratio_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Interest_Coverage_Ratio_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Interest_Coverage_Ratio_Label'] = 'sell'


    '''
    Ben
    '''
    ######################################
    # Calculate Total Shares Repurchased #
    ######################################
    merged_df['Total_Shares_Repurchased'] = merged_df['cshopq']
    
    merged_df['Total_Shares_Repurchased_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Total_Shares_Repurchased_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Total_Shares_Repurchased_Label'] = 'sell'

    # ############################
    # # Calculate In Process R&D #
    # ############################
    # merged_df['In_Process_R_D'] = merged_df['rdipq']
    
    # merged_df['In_Process_R_D_Label'] = 'hold'
    # merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'In_Process_R_D_Label'] = 'buy'
    # merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'In_Process_R_D_Label'] = 'sell'    
    
    ##################################
    # Calculate Capital Expenditures #
    ##################################
    merged_df['Capital_Expenditures'] = merged_df['capxy']
    
    merged_df['Capital_Expenditures_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Capital_Expenditures_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Capital_Expenditures_Label'] = 'sell'
    
    ############################
    # Calculate Cash Dividends #
    ############################
    merged_df['Cash_Dividends'] = merged_df['dvy']
    
    merged_df['Cash_Dividends_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Cash_Dividends_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Cash_Dividends_Label'] = 'sell'

    ##############################
    # Calculate Deferred Revenue #
    ##############################
    merged_df['Deferred_Revenue'] = merged_df['drcq']
    
    merged_df['Deferred_Revenue_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Deferred_Revenue_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Deferred_Revenue_Label'] = 'sell'

    '''
    Jackson
    '''
    #########################
    # Calculate Debt Signal #
    #########################
    merged_df['Debt'] = merged_df['dlcq'] 

    merged_df['Debt_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Debt_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Debt_Label'] = 'sell'

    ###########################################
    # Calculate Debt To Equity Ratio Signal #
    ###########################################
    merged_df['Debt_To_Equity_Ratio'] = merged_df['dlcq']/merged_df['ceqq']

    merged_df['Debt_To_Equity_Ratio_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Debt_To_Equity_Ratio_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Debt_To_Equity_Ratio_Label'] = 'sell'
    
    ####################
    # Calculate EBITDA #
    ####################
    merged_df['EBITDA'] = merged_df['oibdpy'] + merged_df['dpq']

    merged_df['EBITDA_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'EBITDA_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'EBITDA_Label'] = 'sell'

    ##############################
    # Calculate Enterprise Value #
    ##############################
    merged_df['Enterprise_Value'] = merged_df['mkvaltq'] + merged_df['dlttq'] + merged_df['dlcq'] - merged_df['chq']

    merged_df['Enterprise_Value_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Enterprise_Value_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Enterprise_Value_Label'] = 'sell'

    #################################
    # Calculate Enterprise Multiple #
    #################################
    merged_df['Enterprise_Multiple'] = (merged_df['Enterprise_Value'])/(merged_df['EBITDA'])

    merged_df['Enterprise_Multiple_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Enterprise_Multiple_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Enterprise_Multiple_Label'] = 'sell'


    '''
    Rohan
    '''
    ####################
    # Calculate Volume #
    ####################
    merged_df['Volume'] = merged_df['VOL']
    # Categorize into 'Buy', 'Sell', 'Hold' based on percentage thresholds
    buy_threshold_vol = .05  # 5% buy threshold; adjustable
    sell_threshold_vol = -.05  # 5% sell threshold; adjustable
    merged_df.loc[merged_df['Volume'] > buy_threshold_vol, 'Volume_Label'] = 'buy'  
    merged_df.loc[merged_df['Volume'] < sell_threshold_vol, 'Volume_Label'] = 'sell'  

    #####################
    # Calculate EPS New #
    #####################
    # Calculate your signal
    merged_df['EPS_New'] = merged_df['epspiq']

    # Optional: set a new buy_threshold and sell threshold. There's a default one (default_buy_threshold and default_sell_threshold)
    # Optional: Define B/H/S Threshold
    buy_threshold_eps_new = 0.05   # adjustable
    sell_threshold_eps_new = -0.05   # adjustable

    merged_df['EPS_New_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > buy_threshold_eps_new, 'EPS_New_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < sell_threshold_eps_new, 'EPS_New_Label'] = 'sell'

    #######################
    # Calculate P/E Ratio #
    #######################
    # Calculate your signal
    merged_df['PE_Ratio'] = merged_df['PRC']/(merged_df['EPS_New']+1)

    # Optional: set a new buy_threshold and sell threshold. There's a default one (default_buy_threshold and default_sell_threshold)
    # Optional: Define B/H/S Threshold
    buy_threshold_pe_ratio = 0.05   # adjustable
    sell_threshold_pe_ratio = -0.05   # adjustable

    merged_df['PE_Ratio_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > buy_threshold_pe_ratio, 'PE_Ratio_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < sell_threshold_pe_ratio, 'PE_Ratio_Label'] = 'sell'

    #################s/signal_info.py
# signal_label_dict_og, st
    #################
    rsi_period = 14  # Standard RSI period; can adjust
    delta = merged_df['PRC'].diff()  # Difference between current and previous close
    gain = np.where(delta > 0, delta, 0)  # Only positive gains
    loss = np.where(delta < 0, -delta, 0)  # Only losses
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss  # Relative strength
    merged_df['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

    merged_df['RSI_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'RSI_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'RSI_Label'] = 'sell'

    #############################
    # Calculate Bollinger Bands #
    #############################
    merged_df['Rolling_Mean'] = merged_df['PRC'].rolling(window=20).mean()  # 20-day moving average
    merged_df['Rolling_Std'] = merged_df['PRC'].rolling(window=20).std()   # 20-day standard deviation
    merged_df['Upper_Band'] = merged_df['Rolling_Mean'] + (2 * merged_df['Rolling_Std'])  # Upper Bollinger Band
    merged_df['Lower_Band'] = merged_df['Rolling_Mean'] - (2 * merged_df['Rolling_Std'])  # Lower Bollinger Band

    merged_df['Bollinger_Bands_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Bollinger_Bands_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Bollinger_Bands_Label'] = 'sell'

    ###############################
    # Calculate Bollinger Bands alternate #
    ###############################
    merged_df['Bollinger_Bands_alternate_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Bollinger_Bands_alternate_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Bollinger_Bands_alternate_Label'] = 'sell'



    '''
    Ryan
    '''
    #############################
    # Calculate Current Assests #
    #############################
    merged_df['Current_Assets'] = merged_df['actq']

    # Define buy and sell conditions based on price acceleration
    merged_df['Current_Assets_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Current_Assets_Label'] = 'buy'  # Positive acceleration indicates a buy signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Current_Assets_Label'] = 'sell'  # Negative acceleration indicates a sell signal

    #############################
    # Calculate Cash Flow Model #
    #############################
    merged_df['Cash_Flow_Model'] = merged_df['scfq']

    # Define buy and sell conditions based on price acceleration
    merged_df['Cash_Flow_Model_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Cash_Flow_Model_Label'] = 'buy'  # Positive acceleration indicates a buy signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Cash_Flow_Model_Label'] = 'sell'  # Negative acceleration indicates a sell signal


    ##########################
    # Calculate Income Taxes #
    ##########################
    merged_df['Income_Taxes'] = merged_df['txtq']

    # Define buy and sell conditions based on price acceleration
    merged_df['Income_Taxes_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Income_Taxes_Label'] = 'buy'  # Positive acceleration indicates a buy signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Income_Taxes_Label'] = 'sell'  # Negative acceleration indicates a sell signal

    ##########################
    # Calculate Acquisitions #
    ##########################
    merged_df['Acquisitions'] = merged_df['aqcy']

    # Define buy and sell conditions based on price acceleration
    merged_df['Acquisitions_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Acquisitions_Label'] = 'buy'  # Positive acceleration indicates a buy signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Acquisitions_Label'] = 'sell'  # Negative acceleration indicates a sell signal

    ###########################################
    # Calculate Depreciation and Amortization #
    ###########################################
    merged_df['Depreciation_Amortization'] = merged_df['dpy']

    # Define buy and sell conditions based on price acceleration
    merged_df['Depreciation_Amortization_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Depreciation_Amortization_Label'] = 'buy'  # Positive acceleration indicates a buy signal
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Depreciation_Amortization_Label'] = 'sell'  # Negative acceleration indicates a sell signal


    '''
    Ved
    '''
    ##########################
    # Calculate Profit Margin#
    ##########################
    merged_df['Profit_Margin'] = merged_df['niq']/merged_df['revty']
    # threshold_buy = 0.10
    # threshold_sell = 0.02
    # TODO: going forward @Ved don't name them this

    merged_df['Profit_Margin_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Profit_Margin_Label'] = 'buy'  
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Profit_Margin_Label'] = 'sell'  

    # TODO: Ved let's hop on a call and justify this later and rename
    ###################################
    # Calculate Risk-Free Rate Signal #
    ###################################
    merged_df['Risk_Free_Rate'] = pd.to_numeric(merged_df['optrfrq'])
    merged_df['Risk_Adjusted_Return'] = merged_df['RET'] - merged_df['Risk_Free_Rate']

    # Define thresholds
    threshold_buy_rf = 0.02  # Stock outperforms risk-free rate by 2%
    threshold_sell_rf = -0.02  # Stock underperforms risk-free rate by 2%

    # Define buy and sell conditions based on comparison to risk-free rate
    merged_df['Risk_Adjusted_Return_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > threshold_buy_rf, 'Risk_Adjusted_Return_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < threshold_sell_rf, 'Risk_Adjusted_Return_Label'] = 'sell'

    ####################################
    # Calculate Total Invested Capital #
    ####################################
    merged_df['Total_Invested_Capital'] = merged_df['ltq'] + merged_df['teqq']

    # # Define thresholds (you may need to adjust these based on your specific needs)
    # threshold_buy_tic = merged_df['Total_Invested_Capital'].quantile(0.75)  # Top 25%
    # threshold_sell_tic = merged_df['Total_Invested_Capital'].quantile(0.25)  # Bottom 25%

    # Define buy and sell conditions based on Total Invested Capital
    merged_df['Total_Invested_Capital_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Total_Invested_Capital_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Total_Invested_Capital_Label'] = 'sell'

    ####################################
    # Calculate Net Receivables Signal #
    ####################################
    # Assuming 'rectr' is the column for Net Receivables
    merged_df['Net_Receivables'] = merged_df['rectrq']

    # # Define thresholds (you may need to adjust these based on your specific needs)
    # threshold_buy_nr = merged_df['Net_Receivables'].quantile(0.75)  # Top 25%
    # threshold_sell_nr = merged_df['Net_Receivables'].quantile(0.25)  # Bottom 25%

    # Define buy and sell conditions based on Net Receivables
    merged_df['Net_Receivables_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'Net_Receivables_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'Net_Receivables_Label'] = 'sell'

    #########################
    # Calculate ROIC Signal #
    #########################
    # Ensure we don't divide by zero
    merged_df['ROIC'] = merged_df['oiadpq'] / merged_df['icaptq'].replace(0, np.nan)

    # # Define thresholds (you may need to adjust these based on your specific needs)
    # threshold_buy_roic = merged_df['ROIC'].quantile(0.75)  # Top 25%
    # threshold_sell_roic = merged_df['ROIC'].quantile(0.25)  # Bottom 25%

    # Define buy and sell conditions based on ROIC
    merged_df['ROIC_Label'] = 'hold'
    merged_df.loc[merged_df['Future_Price_Change'] > default_buy_threshold, 'ROIC_Label'] = 'buy'
    merged_df.loc[merged_df['Future_Price_Change'] < default_sell_threshold, 'ROIC_Label'] = 'sell'

    '''
    Done
    '''
    ##########################################
    ### Below this box drop excess columns ###
    ##########################################
    to_drop = list(set(list(merged_df.columns))-set(general_columns_list))
    to_drop = list(set(to_drop)-set(signal_list))
    to_drop = list(set(to_drop)-set(signal_label_list))
    merged_df.drop(columns = to_drop, inplace = True)
    for col in merged_df.columns:
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan)
    print(merged_df)
    return merged_df

#########################################################################################
#################################### PORTFOLIO SETUP ####################################
#########################################################################################

# Function that spits out different portfolios to try different things out
def portfolio_types(merged_data, permnos, signal_label_dict):
    ##########
    # Format #
    ##########
    # pn_keys = [[label1, label2, label3], [label4, label5], ...]
    # pn_permnos = [[permno1, permno2], [permno3], ...]
    ### Remember to add to portfolio_permnos_list and portfolio_keys_list

    # TODO: Try different portfolios, first by grouping several similar keys together
    # TODO: Try grouping permnos, first by grouping by various "sets" of industries

    p0_keys = [['Gen_Label']]
    p0_permnos = [sorted(merged_data.loc[:,'permno'].unique())]
    
    p1_keys_to_exclude = ['Gen_Label']
    p1_keys = [[key] for key in signal_label_dict.keys() if key not in p1_keys_to_exclude]
    p1_permnos = [sorted(merged_data.loc[:,'permno'].unique())]
    
    p2_keys = [['Gen_Label']]
    p2_permnos = permnos

    p3_keys_to_exclude = ['Gen_Label']
    p3_keys = [[key] for key in signal_label_dict.keys() if key not in p3_keys_to_exclude]
    p3_permnos = permnos

    permnos_by_naics = group_by_naics(merged_data)

    portfolio_permnos_list = [p0_permnos, p1_permnos, p2_permnos, p3_permnos]
    portfolio_keys_list = [p0_keys, p1_keys, p2_keys, p3_keys]
    portfolio_interpretatbility_list = portfolio_interpretability_determination(portfolio_permnos_list, portfolio_keys_list)
    return portfolio_permnos_list, portfolio_keys_list, portfolio_interpretatbility_list

# This needs to be reworked if we ever start using the same signals in multiple models
# Function that looks at our permnos and keys and see how granular they are. The more granular, the more interpretable.
def portfolio_interpretability_determination(portfolio_permnos_list, portfolio_keys_list):
    max_permno_len = 0
    max_key_len = 0
    for p_ind in range(len(portfolio_keys_list[0:4])):
        max_permno_len = max(max_permno_len, len(flatten(portfolio_permnos_list[p_ind])))
        max_key_len = max(max_key_len, len(flatten(portfolio_keys_list[p_ind])))
    portfolio_interpretatbility_list = []
    for p_ind in range(len(portfolio_permnos_list)):
        permno_score = len(portfolio_permnos_list[p_ind])/max_permno_len
        key_score = len(portfolio_keys_list[p_ind])/max_key_len
        portfolio_interpretatbility_list.append(permno_score*key_score)
    return portfolio_interpretatbility_list

# The following two functions allow grouping permnos by naics
def group_by_naics(merged_df):
    grouped_df = merged_df.groupby('permno')['naics'].apply(most_frequent_naics).reset_index()
    result = grouped_df.groupby('naics')['permno'].apply(list).tolist()
    return result

# This just makes sure that the most common naics for each permno is used.
def most_frequent_naics(group):
    counter = Counter(group)
    return counter.most_common(1)[0][0]

# Imports
import pandas as pd
import matplotlib as plt
import datetime as dt

#########################
####### Functions #######
#########################

### Extract trading dates
# Extracts trading dates from the input file
# It does this by looking in merged.dta, then sorting the unique end of month dates

def extract_dates(path='Inputs/Created/merged.dta'):
    try:
        df = pd.read_stata(path)
    except:
        df = pd.read_stata('../' + path)
    
    # Convert the 'date' column to datetime format
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], format="%Y-%m-%d")
    # Extract the year and month as a new column
    df['year_month'] = df['date'].dt.to_period('M')
    # Get the last date for each year-month group
    last_dates = df.groupby('year_month')['date'].max()
    
    # Sort the result
    sorted_last_dates = sorted(last_dates)

    return sorted_last_dates


### Create plot comparing alpha and interpretability score
def plot_alpha_interpretability(portfolio_list, portfolio_interpretability_list):
    alpha_list = []
    for p in portfolio_list:
        alpha_list.append(p.ann_alpha)
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_interpretability_list, alpha_list, marker='o')
    plt.title('Alpha Interpretability')
    plt.xlabel('Portfolio')
    plt.ylabel('Interpretability')
    plt.grid(True)
    plt.show()


########################################
####### Inputs & Constants below #######
########################################

### Sandbox options
# -------------------------------------------------------------------------------------------------
# Setup for Google Colab (currently not finished)
# -------------------------------------------------------------------------------------------------
IN_COLAB = False

### Constants
# -------------------------------------------------------------------------------------------------
# Date/Time Initialization
# -------------------------------------------------------------------------------------------------
if not('missing_signal' in locals() or 'missing_signal' in globals()):
    time_date = str(dt.datetime.now().strftime('%Y_%m_%d_%H_%M'))

# -------------------------------------------------------------------------------------------------
# In-sample Testing Split Configuration
# -------------------------------------------------------------------------------------------------
is_test_train_split = .68
is_test_split_date = '2012-01-01' # 67% split

# Have only one of these for the real deal
# test_train_split = .7
# split_date = '2012-01-01'
test_train_split = .802
split_date = '2015-01-01' # 80% split

# -------------------------------------------------------------------------------------------------
# File Paths for Input Data
# -------------------------------------------------------------------------------------------------
market_data_path = "Inputs/Downloaded/mkt_rf_daily.csv"
# Real data
price_data_path = "Inputs/Downloaded/price_data_1991-2020.csv"
fund_data_path = "Inputs/Downloaded/Fundamentals_1991-2020.dta"
merged_data_path = "Inputs/Created/merged.dta"

# Test data
price_data_test_path = "Inputs/Downloaded/price_data_test.csv"
fund_data_test_path = "Inputs/Downloaded/Fundamentals_test.dta"
merged_data_test_path = "Inputs/Created/merged_test.dta"

# Full data
price_data_full_path = "Inputs/Downloaded/price_data_full.csv"
fund_data_full_path = "Inputs/Downloaded/Fundamentals_full.dta"
merged_data_full_path = "Inputs/Created/merged_full.dta"

conversion_path = 'Inputs/Downloaded/gvkey_permno_conversion.dta'

# -------------------------------------------------------------------------------------------------
# Other
# -------------------------------------------------------------------------------------------------
# Currently based on days. small_ret = 1, big_ret = 12 if monthly data
small_ret = 21
big_ret = 252
# How many stocks in the dataset are needed to trade
min_stocks_available = 1
# Create mapping
map_to_dec = {-1:'sell', 0:'hold', 1:'buy'} # mapping number to decision
map_to_num = {'sell':-1, 'hold':0, 'buy':1} # mapping decision to number
# Because accounting data is oftentimes released EOD, we can't use that data to trade on the same day
min_accounting_lag = 1

## Determine if this should run on test, normal, or full data, and then importing it
# Check if 'is_test' is already defined. If not, ask the user whether it's a test or not
if not('is_test' in locals() or 'is_test' in globals()):
    test_added = ''
    full_added = ''
    try:
        with open('Inputs/AlwaysTest.txt', 'r') as file:
            always_test = bool(file.read)
    except FileNotFoundError:
        raise FileNotFoundError("There is no AlwaysTest.txt file.\nCopy and paste AlwaysTest.template.txt into the same directory and rename it to AlwaysTest.txt.\nThen edit the variable as you wish.")
    if always_test:
        is_test = True
        is_full = False
    else:
        while True:
            is_test_input = input('Is this a test? (Enter True or False) ') # User prompt
            if is_test_input == "True":
                is_test = True
                break
            if is_test_input == "False":
                is_test = False
                break
            else:
                print('Try again.')
        
        if not is_test:
            while True:
                is_full_input = input('Is this the full data? (Enter True or False) ') # User prompt
                if is_full_input == "True":
                    is_full = True
                    break
                if is_full_input == "False":
                    is_full = False
                    break
                else:
                    print('Try again.')
        else:
            is_full = False

if (not is_test) and (not is_full):
    price_data = pd.read_csv(price_data_path)
    fund_data = pd.read_stata(fund_data_path)
    m_path = merged_data_path
elif (not is_test) and is_full:
    full_added = '_full'
    price_data = pd.read_csv(price_data_full_path, iterator=True)
    fund_data = pd.read_stata(fund_data_full_path, iterator=True)
    m_path = merged_data_full_path
else:
    test_added = '_test'
    price_data = pd.read_csv(price_data_test_path)
    fund_data = pd.read_stata(fund_data_test_path)
    m_path = merged_data_test_path

# -------------------------------------------------------------------------------------------------
# Loading and cleaning merged data
# -------------------------------------------------------------------------------------------------
# Run initial_data_processor.ipynb if merged.dta or merged_test.dta is missing
try:
    # from pandas.io.stata import StataReader
    # reader = StataReader(m_path)

    merged_data = pd.read_stata(m_path)
    # for testing purposes
    merged_data.fillna(0, inplace=True)

    try:
        merged_data.drop('index', axis = 1, inplace=True)
    except:
        aaaa = 0 # If 'index' column doesn't exist, do nothing
except:
    print('running data processor')
    %run Background_Scripts/initial_data_processor.ipynb
    merged_data = pd.read_stata(m_path)
     # for testing purposes
    merged_data.fillna(0, inplace=True)
    try:
        merged_data.drop('index', axis = 1, inplace=True)
    except:
        aaaa = 0

##############################
### Running signal_info.py ###
##############################
##### Getting user adjustments from signal_info.py
##### But first has to check if this is being run from MarginalAnalysis, which would cause 'missing_signal' to exist as a variable already; if not then missing_signal must be defined to put the Outputs in the right folder
##### WARNING: may need to restart kernel if TradingAlgorithms was run before; currently untested
if not('missing_signal' in locals() or 'missing_signal' in globals()):
    missing_signal = -1
    %run Background_Scripts/signal_info.py
    signal_label_dict_og, strategy_info = create_signal_label_dict(merged_data)
    signal_label_dict = signal_label_dict_og.copy()
    # What has been defined: paths to various inputs, date splitting constants, signal_label_dict, current_signals, and num_signals

# -------------------------------------------------------------------------------------------------
# Date setup
# -------------------------------------------------------------------------------------------------
merged_data['date_old'] = pd.to_datetime(merged_data['date'])
ref_date = merged_data['date_old'].min()
merged_data['date'] = (merged_data['date_old'] - ref_date).dt.days

# Create a sorted list of unique dates from the 'merged_data'
udates = sorted(merged_data.loc[:,'date'].unique())
# Split the merged data into training and test sets based on the split date (or test split date if it's a test run)
if type(split_date) == type('amogus'):
    split_date = dt.datetime.strptime(split_date, '%Y-%m-%d')
else:
    print(split_date)
split_date_numeric = (split_date - ref_date).days
if not is_test:
    merged_data_train = merged_data[merged_data['date'] < split_date_numeric]
    merged_data_test = merged_data[merged_data['date'] >= split_date_numeric]
else:
    is_test_split_date = dt.datetime.strptime(is_test_split_date, '%Y-%m-%d')
    is_test_split_date_numeric = (is_test_split_date - ref_date).days
    merged_data_train = merged_data[merged_data['date'] < is_test_split_date_numeric]
    merged_data_test = merged_data[merged_data['date'] >= is_test_split_date_numeric]

### Post import constants
# -------------------------------------------------------------------------------------------------
# Create a dictionary for fast access to merged data by date
# -------------------------------------------------------------------------------------------------
# Create a dictionary mapping each date to a list of the corresponding 'permno' rows (stocks) and their associated signals
columns_to_keep = list(merged_data_test.columns)
columns_to_keep.remove('date')
columns_to_keep.remove('permno')

# grouped = merged_data_test.groupby('date').apply(lambda x: x.set_index('permno')[columns_to_keep].apply(lambda row: row.tolist(), axis=1).to_dict()).reset_index(name='items')
# grouped = merged_data_test.groupby('date').apply(lambda x: x.drop(columns=['date']).set_index('permno')[columns_to_keep].apply(lambda row: row.tolist(), axis=1).to_dict()).reset_index(name='items')
grouped = merged_data_test.groupby('date').apply(lambda x: x.set_index('permno')[columns_to_keep].apply(lambda row: row.tolist(), axis=1).to_dict(),include_groups=False).reset_index(name='items') # Exclude 'date' from the grouped data

# Convert to dictionary
merged_data_dict = dict(zip(grouped['date'], grouped['items']))

# grouped = merged_data_test.groupby('date').apply(lambda x: x.set_index('permno')[columns_to_keep].apply(lambda row: row.tolist(), axis=1).to_dict()).reset_index(drop=True)
# grouped.columns = ['date', 'items']  # Rename columns to match your intended output


### Fitting original models- the 'all' column is used for creating models that use all the data (either stocks or shares)
stocks = sorted(merged_data['permno'].unique())
num_stocks = len(stocks)
stocks_og = stocks.copy()
stocks = np.append(stocks, 'all_stocks')

# Columns for buys & sells dfs
buys_sells_columns = (['permno','quantity'] + list(signal_label_dict.keys()))

# -------------------------------------------------------------------------------------------------
# End of Month Trading Dates
# -------------------------------------------------------------------------------------------------
# %run Background_Scripts/EOM_Trading_Dates.ipynb
if not is_test:
    EOM_dates = extract_dates(m_path)[:-1]
    old_trading_dates = EOM_dates[round(test_train_split*len(EOM_dates)-1):]
    old_trading_dates = pd.to_datetime(old_trading_dates)
    trading_dates = (old_trading_dates - ref_date).days
else:
    EOM_dates = extract_dates(m_path)[:-1]
    old_trading_dates = EOM_dates[round(is_test_train_split*len(EOM_dates)-1):]
    old_trading_dates = pd.to_datetime(old_trading_dates)
    trading_dates = (old_trading_dates - ref_date).days

next_day_map = {}
for i in range(len(trading_dates) - 1):
    next_day_map[trading_dates[i]] = trading_dates[i + 1]

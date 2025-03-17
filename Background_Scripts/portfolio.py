import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Portfolio():
    
    # def intialize(self):
    #     # Initialize with the path to your CSV file
    #     self.data = None
    #     self.pivot_data = None
    #     self.fig = None
    #     self.ax = None
        
    #self.output_folder_path / "pdp_plots" / f"{feature}_{st}.png"
        
        
    
    def __init__(
            self, signal_set = [[]], stock_set = [[]], strategy = 0, missing_signal = -1,
            merged_data = pd.DataFrame(), merged_data_train = pd.DataFrame(), merged_data_test = pd.DataFrame(),
            extra_info = False
            ):
        assert type(signal_set) == list
        assert type(stock_set) == list

        self.signal_set = signal_set
        self.stock_set = stock_set

        #sorting testing, training, merged data by date & permno
        self.merged_data = merged_data.sort_values(by=['date', 'permno'])
        self.data_train = merged_data_train.sort_values(by=['date', 'permno'])
        self.data_test = merged_data_test.sort_values(by=['date', 'permno'])

        self.model_lol = self.model_creation()

        #stores current holdings & previous trades
        self.current_holdings = pd.DataFrame(columns=['permno', 'per_dollar_amount'])
        self.prev_trades = pd.DataFrame(columns=['date_open', 'date_close', 'permno', 'per_dollar_amount_open', 'per_dollar_amount_close', 'num_shares', 'PRC_open', 'PRC_close', 'P&L'])
        
        self.strategy_info = strategy_info
        self.missing_signal = missing_signal
        self.output_folder_path = self.make_backtest_path(self.strategy_info['output folder name'], strategy)
        
        #stores trades & net asset value
        self.trades = pd.DataFrame()
        self.nav_df = pd.DataFrame()
        self.ann_alpha = 0
        self.extra_info = extra_info


    def model_creation(self):
        import copy
        """
        creating label encoder to convert text/categorical/string labels into numbers
        Iterates through each signal set & extracting feats from the signal label dictionary
        & creates a random forest classifier for each stock set

        returns list of trained models
        """
        label_encoder = LabelEncoder()
        model_lol = []
        for si in self.signal_set:
            model_list = []
            features = []
            for key in si:
                features = features + signal_label_dict[key]

            X_train = self.data_train[['date', 'permno'] + features]
            for st in self.stock_set:
                subset_X_train = X_train[X_train['permno'].isin(st)]

                # TODO: Check to see if necessary to add in extra 'permnos'
                # edited_data = self.data_train[self.data_train['permno'].isin(st)][['date', 'permno'] + si + features]
                # # Convert 'date' column to datetime if not already
                # edited_data['date'] = pd.to_datetime(edited_data['date'])
                # # Option 2: Convert to days since a particular reference date (e.g., the first date in your dataset)
                # ref_date = edited_data['date'].min()
                # edited_data['date_numeric'] = (edited_data['date'] - ref_date).dt.days
                # X_train = edited_data[['date_numeric', 'permno'] + features]
                
                # I think the following lines can be made more efficient
                y = self.data_train[self.data_train['permno'].isin(st)]
                y_train = label_encoder.fit_transform(y[si].values.ravel()  )

                curr_model = RandomForestClassifier(n_estimators=50, random_state=42)#, warm_start=True)
                # TODO: @Ben
                ##########format for adding more trees/incremental learning ##############
                #model.set_params(n_estimators=?)if 50 trees set to 60 for 10 more trees
                #model.fit(X_new, y_new)
                ###############################################
                #look into random states
                curr_model.fit(subset_X_train, y_train)
                model_list.append(curr_model)
            model_lol.append(model_list)

        return model_lol 

    def model_access(self, og_X_test):
        """
        Make prediction using trained models & test deta
        Returns res, a list of predictions for all the stocks

        Iterates through the signals to make predictions & extracts feautures based on current signal set
        Sums the predictions then converts the sum to buy, sell, or hold signals
        """
        edited_data = og_X_test[:][:]
        # Convert 'date' column to datetime if not already for date based predictions
        edited_data['date_old'] = pd.to_datetime(edited_data['date'])
        # Option 2: Convert to days since a particular reference date (e.g., the first date in your dataset)
        internal_ref_date = edited_data['date_old'].min()
        edited_data['date'] = (edited_data['date_old'] - internal_ref_date).dt.days #changing dates to something machine can read
        curr_pred = []
        i = 0

        for si in self.signal_set:
            
            features = []
            for key in si:
                features = features + signal_label_dict[key]
            X_test = edited_data[['date', 'permno'] + features]
            sig_pred = []
            j = 0
            for st in self.stock_set:
                subset_X_test = X_test[X_test['permno'].isin(st)]
                if len(subset_X_test) == 0:
                    sig_pred.append(np.array([1]*len(si))/len(self.signal_set))
                else:
                    curr_model = self.model_lol[i][j]
                    pred = curr_model.predict(subset_X_test)
                    sig_pred.append(pred/len(self.signal_set))
                j += 1
 
            curr_pred.append(sig_pred)
            i += 1

        summed_array = [0]*num_stocks
        ind = 0
        for i in curr_pred:
            for j in i:
                for k in j:
                    summed_array[ind%num_stocks] += float(k)
                    ind += 1
        res = [1 if x > 4/3 else -1 if x < 2/3 else 0 for x in summed_array]
        return res # This prediction is for all stocks for that day
    
    def compute_trades(self, start_date = -1, end_date = -1):
        """
        Iterates through all trading dates (from start to finish) & determines buy,sell,hold based on predicitons which are then
        stored in prev_trades

        Access the predictions for the current date, check for non-zero predictions, see if its buy or sell
        Then interate through the predictions at that date, if prediction is buy, create a row for trade, if prediction is sell create a row for sell in self.prev_trades
        If prediction is 0, skip

        Then add the trade to the prev_trades
        """
        # If computing all trades in the testing dataset
        if start_date == -1 and end_date == -1:
            total_test_set = self.data_test
        else:
            total_test_set = self.data_test[(self.data_test['date'] >= start_date) & (self.data_test['date'] < end_date)]
        for date in trading_dates:
            curr_test_set = total_test_set[total_test_set['date']==date]
            curr_pred = self.model_access(curr_test_set)
            if any(x != 0 for x in curr_pred):
                num_buys = sum(1 for x in curr_pred if x > 0)
                num_sells = sum(1 for x in curr_pred if x < 0)
                
                for permno_ind in range(len(curr_pred)):
                    if curr_pred[permno_ind] > 0 and num_buys > 0:
                        new_row = {'date_open': date, 'permno':stocks_og[permno_ind], 'per_dollar_amount_open':1.0/num_buys}
                    elif curr_pred[permno_ind] < 0 and num_sells > 0:
                        new_row = {'date_open': date, 'permno':stocks_og[permno_ind], 'per_dollar_amount_open':-1.0/num_sells*.5}
                    else:
                        continue
                    self.prev_trades.loc[len(self.prev_trades)] = new_row
        self.trades = self.prev_trades
        
    def execute_trades(self, starting_nav=100):
        """
        Calculated num shares that are bought and sold, profit & loss of the trades
        Calculates net asset value of entire portfolio over period

        Iterates through each trading date, gets the trades that are closed at date.
        A. Then gets #shares (open dollar amount / opening P)
        B. gets closing dollar amount (#shares * closing P)
        C. gets profit or loss (A-B)

        Updates the net asset value by adding profit/loss of the current_df (C)
        """
        # Ensure the dataframes are sorted by date (ascending)
        prices_df = self.data_test[['date', 'permno', 'PRC']].copy()
        trades_df = self.prev_trades[:][:]
        new_trades_df = pd.merge(trades_df, prices_df, left_on=['date_open', 'permno'], right_on=['date', 'permno'], how='left')
        new_trades_df['PRC_open'] = new_trades_df['PRC']
        new_trades_df.drop(['PRC', 'date'], axis=1, inplace=True)
        new_trades_df['date_close'] = new_trades_df['date_open'].map(next_day_map)
        new_trades_df = pd.merge(new_trades_df, prices_df, left_on=['date_close', 'permno'], right_on=['date', 'permno'], how='left')
        new_trades_df['PRC_close'] = new_trades_df['PRC']
        new_trades_df.drop(['PRC', 'date'], axis=1, inplace=True)

        nav = starting_nav
        nav_list = [nav]
        nav_df = pd.DataFrame(columns=['date', 'nav']).set_index('date')
        for date in trading_dates:
            current_df = new_trades_df[new_trades_df['date_close']==date]

            current_df.loc[:,'num_shares'] = current_df.loc[:,'per_dollar_amount_open']/current_df.loc[:,'PRC_open']*nav_list[-1]
            current_df.loc[:,'per_dollar_amount_close'] = current_df.loc[:,'PRC_close']*current_df.loc[:,'num_shares']/nav_list[-1]
            current_df.loc[:,'P&L'] = (current_df.loc[:,'per_dollar_amount_close'] - current_df.loc[:,'per_dollar_amount_open'])*nav_list[-1]
            # then figure out quantities and shit

            nav_list.append(nav_list[-1] + sum(current_df['P&L']))
            nav_df.loc[date,'nav'] = nav_list[-1]

        self.nav_df = nav_df
        # Plot the NAV over time
        # self.nav_plot()
        # return nav_df

    def output_stats(self):               
        """
        Outputs statistics and plots to files
        """
        # Output full csvs for account_history and trades
        self.nav_df.to_csv( self.output_folder_path / 'account_history.csv' )
        self.trades.to_csv( self.output_folder_path / 'trades.csv' )
        
        # Compute and output main stats, which will go in this dictionary
        backtest_stats = dict()
        
        # Get the return series from the account history, % change in Net Asset Value of portfolio
        ret_series = self.nav_df['nav'].astype(float).pct_change().dropna().reset_index(drop=True)
        
        # Number of periods per year for annualization
        N = self.strategy_info['periods per year']
        
        # finds mean & geometric mean of returns
        backtest_stats['arith mean'] = ret_series.mean()
        backtest_stats['arith mean (ann)'] = ret_series.mean()*N
        backtest_stats['geomean mean'] = stats.gmean(1+ret_series)-1
        backtest_stats['geomean mean (ann)'] = stats.gmean(1+ret_series)**N-1
        
        # finds risks/volitility, max & average & current drawndowns
        backtest_stats['sigma'] = ret_series.std()
        backtest_stats['sigma (ann)'] = ret_series.std()*(N**0.5)
        drawdown = 1-self.nav_df.loc[:,'nav'] / self.nav_df.loc[:,'nav'].cummax()
        backtest_stats['avg drawdown'] = drawdown.mean()
        backtest_stats['max drawdown'] = drawdown.max()
        
        #reads market data from CSV file market_data_path to get alpha/beta, %change in market index, %change in RFR
        mkt_rf_df = pd.read_csv(market_data_path)
        mkt_rf_df.loc[:,'date1'] = mkt_rf_df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        mkt_rf_df['date'] = mkt_rf_df['date1']
        del mkt_rf_df['date1']
        mkt_rf_df['int_date'] = (mkt_rf_df['date'] - ref_date).dt.days
        mkt_rf_df = mkt_rf_df.loc[mkt_rf_df.loc[:,'int_date'].isin(self.nav_df.index),:]
        mkt_series = mkt_rf_df.loc[:,'mkt_index'].pct_change().dropna().reset_index(drop=True)
        rf_series = mkt_rf_df.loc[:,'rf_index'].pct_change().dropna().reset_index(drop=True)
        
        # Set up left- and right-hand sides for regressions
        lhs = ret_series
        rhs = sm.add_constant( mkt_series - rf_series )
        
        # Run the ordinary least squares regression & fits it to data
        model = sm.OLS(lhs,rhs) 
        results = model.fit()
        
        # extract alpha and beta
        backtest_stats['alpha'] = results.params['const']
        backtest_stats['alpha SE'] = results.bse['const']
        backtest_stats['alpha (ann)'] = results.params['const']*N
        backtest_stats['alpha SE (ann)'] = results.bse['const']*N
        backtest_stats['beta'] = results.params[0]
        backtest_stats['betaSE'] = results.bse[0]        
        
        # extract ratios: Sharpe & information ratio       
        backtest_stats['Sharpe ratio'] = backtest_stats['arith mean'] / backtest_stats['sigma']
        backtest_stats['Sharpe ratio (ann)'] = backtest_stats['arith mean (ann)'] / backtest_stats['sigma (ann)']
        backtest_stats['information ratio'] = backtest_stats['alpha'] / backtest_stats['sigma']
        backtest_stats['information ratio (ann)'] = backtest_stats['alpha (ann)'] / backtest_stats['sigma (ann)']
        
        # Turn into a dataframe and output as csv
        pd.DataFrame.from_dict(data=backtest_stats, orient='index').to_csv(self.output_folder_path / 'backtest_stats.csv')

        self.ann_alpha = backtest_stats['alpha (ann)']
        
        # generates & saves plot of net asset value history
        self.nav_plot()
        if self.extra_info:
            self.pdp_plot()
            self.heatmap()
        
    def print_nav_plot(self):
        """
        Make and display the cumulative NAV history plot
        """
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.nav_df.index, self.nav_df['nav'], label='NAV', color='blue')
        plt.title('Net Asset Value (NAV) Over Time')
        plt.xlabel('Date')
        plt.ylabel('NAV ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()
        
    def nav_plot(self):
        plt.ioff()
        """
        Make and display the cumulative NAV history plot
        """
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.nav_df.index, self.nav_df['nav'], label='NAV', color='blue')
        plt.title('Net Asset Value (NAV) Over Time')
        plt.xlabel('Date')
        plt.ylabel('NAV ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        # Save in our folder
        plt.savefig(self.output_folder_path / 'nav_plot.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        # plt.show()


        # plt.style.use('default')  # set plot style to 'default' 

        # # Standard set-up for a matplotlib plot
        # fig = plt.figure() # this will automatically show the most updated version of the plot after you run the cell, so no need to have a "fig" line later again
        # ax = fig.add_subplot(1,1,1)
        # fig.suptitle(list_of_models[i])
        # plotdates = self.portfolio_db.account_history_df.loc[:,'datetime']
        # ax.plot(plotdates, self.portfolio_db.account_history_df.loc[:,'nav'] / self.portfolio_db.account_history_df.loc[:,'nav'].iloc[0]) 
        
        # # Customize the plot properties
        # ax.set_xlabel('Date')
        # ax.set_yscale('log')
        # ax.yaxis.set_major_locator( ticker.LogLocator(base=2) )        
        # ax.yaxis.set_minor_locator( ticker.LogLocator(base=2, subs=[1.25, 1.5, 1.75]))
        # ax.yaxis.set_major_formatter( ticker.ScalarFormatter() )
        # ax.yaxis.set_minor_formatter( ticker.NullFormatter() )
        # ax.set_ylabel(self.strategy_info['plot descriptor'])
        # ax.set_xlim(plotdates.min(), plotdates.max())
        # ax.xaxis.set_major_formatter( mdates.DateFormatter('%Y'))  # format the tick labels using years
        
    def make_backtest_path(self, output_folder, strategy=0):
        """
        Create the path object for the folder used to store the backtest output
        The folder name will look like YYYYmmddHHMm/stratname_strategy/Num_missing_signal
        Folder names will have unique output name
        """
        output_path = Path(output_folder) 
        
        # This is the output name we want
        folder_name = '{0}_{1}_{2}'.format(self.strategy_info['brief descriptor'], 'missing', missing_signal)
        self.output_folder_path = Path(output_path / time_date / folder_name / str(strategy))  
    
        # It's possible the folder already exists, if it does add a ' (1)' or ' (2)' or however high we need to go
        # This follows the Windows convention for duplicate files
        num_attempts = 1
        while( self.output_folder_path.exists() ):
            folder_name_conflict = folder_name + ' ({0})'.format(num_attempts)
            self.output_folder_path = Path(output_path / time_date / folder_name_conflict / str(strategy))  
            num_attempts += 1
            
        self.output_folder_path.mkdir(parents=True)
        
        return self.output_folder_path
    
    def pdp_plot(self):
        label_encoder = LabelEncoder()
        features = []
        for si in self.signal_set:
            for key in si:
                features = features + signal_label_dict[key]
                
        # Calculate the grid dimensions
        n_plots = len(features) * len(self.stock_set)
        n_cols = 4  # Changed to 4 columns
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create a single figure - adjusted width for 4 columns
        
        X_train = self.data_train[['date', 'permno'] + features]
        plot_idx = 1
        os.makedirs(self.output_folder_path/"pdp_plots", exist_ok=True)#makes folder

        for feature in features:
            for st in self.stock_set:
                # Create subplot and get the axis
                plt.figure(figsize=(8,6))
                
                subset_X_train = X_train[X_train['permno'].isin(st)]
                y = self.data_train[self.data_train['permno'].isin(st)]
                y_train = label_encoder.fit_transform(y[si])
                
                curr_model = RandomForestClassifier(n_estimators=50, random_state=42, warm_start=True)
                targetclass = 0
                curr_model.fit(subset_X_train, y_train)
                
                # Create PDP plot
                PartialDependenceDisplay.from_estimator(curr_model,subset_X_train,features=[feature],grid_resolution=50,target=targetclass,line_kw={'color': 'blue'},kind='average')
                
                # Add title to subplot
                plt.title(f'{feature} for {st}', pad=15)  # Reduced padding slightly
                plt.savefig(self.output_folder_path / "pdp_plots" / f"{feature}_{st}.png" , bbox_inches='tight')#puts pic in folder
                plt.close()    
            
        self.output_folder_path / "pdp_plots" / f"{feature}_{st}.png"
    
    def heatmap(self):
        # Read the CSV file
        data = pd.read_csv('HeatmapData1.csv')

        # Ensure the Percentage Change column is formatted correctly
        data['Percentage Change'] = data['Percentage Change'].str.replace('%', '', regex=True).astype(float)

        # Pivot the data for heatmap structure
        pivot_data = data.pivot(index='Signal/Factor', columns='Model', values='Percentage Change')
        pivot_data = pivot_data.fillna(0)  # Replace NaN values with 0

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, center=0, ax=ax)

        # Modify text labels to include '%'
        for text in ax.texts:
            text.set_text(f"{float(text.get_text()):.2f}%")

        # Titles and labels
        ax.set_title("Signal Performance Across Models")
        ax.set_xlabel("Models")
        ax.set_ylabel("Signals")

        # Define the output file path
        output_path = self.output_folder_path / "heatmap_plot.png"
        
        # Save the plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory

        return output_path  # Return the saved file path for reference

# Delete when done

# HeatmapGen = Portfolio()
# heatmap_path = HeatmapGen.heatmap(output_folder='Outputs/202502041649/5')  # Use the dynamically created folder
# print(f"Heatmap saved at: {heatmap_path}")
    
    # ##png folder location
    # def output_stats(self, output_folder='Outputs/202502041649/5'):
    #     self.fig.savefig(f"{output_folder}/signal_heatmap.png")




# class HeatmapGenerator:
#     def __init__(self, data_path, output_folder="/Users/ryansajwan/GitHub/ML-Interpretability/Outputs/202502041649/5"):
#         self.data_path = data_path
#         self.output_folder = output_folder
#         self.heatmap_plot = None #initialize heatmap_plot.
    
#     def pdp_plot(self):
#         label_encoder = LabelEncoder()
#         features = []
#         for si in self.signal_set:
#             for key in si:
#                 features = features + signal_label_dict[key]
                
#         # Calculate the grid dimensions
#         n_plots = len(features) * len(self.stock_set)
#         n_cols = 4  # Changed to 4 columns
#         n_rows = (n_plots + n_cols - 1) // n_cols
        
#         # Create a single figure - adjusted width for 4 columns
#         fig = plt.figure(figsize=(20, 4 * n_rows))  # Increased width, slightly reduced height per row
        
#         X_train = self.data_train[['date', 'permno'] + features]
#         plot_idx = 1
        
#         for feature in features:
#             for st in self.stock_set:
#                 # Create subplot and get the axis
#                 ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                
#                 subset_X_train = X_train[X_train['permno'].isin(st)]
#                 y = self.data_train[self.data_train['permno'].isin(st)]
#                 y_train = label_encoder.fit_transform(y[si])
                
#                 curr_model = RandomForestClassifier(n_estimators=50, random_state=42, warm_start=True)
#                 targetclass = 0
#                 curr_model.fit(subset_X_train, y_train)
                
#                 # Create PDP plot
#                 disp = PartialDependenceDisplay.from_estimator(curr_model,subset_X_train,features=[feature],grid_resolution=50,target=targetclass,ax=ax,line_kw={'color': 'blue'},kind='average')
                
#                 # Add title to subplot
#                 ax.set_title(f'{feature} for {st}', pad=15)  # Reduced padding slightly
#                 plot_idx += 1
        
#         # Adjusted spacing for 4 columns
#         plt.subplots_adjust(hspace=0.35, wspace=0.2)  # Slightly reduced spacing
#         plt.tight_layout(pad=1.5)  # Reduced padding slightly
#         plt.show()
    
#     def intialize(self, csv_path):
#         # Initialize with the path to your CSV file
#         self.csv_path = csv_path
#         self.data = None
#         self.pivot_data = None
#         self.fig = None
#         self.ax = None
        
#     def heatmap(self):
#         # Load data if not already loaded
#         self.data = pd.read_csv(self.csv_path)
        
#         #Heatmap code formatting/creating
#         self.data['Percentage Change'] = self.data['Percentage Change'].str.replace('%', '', regex=True).astype(float)
#         self.pivot_data = self.data.pivot(index='Signal/Factor', columns='Model', values='Percentage Change')
#         self.pivot_data = self.pivot_data.fillna(0)
        
#         # Create the figure and axis objects
#         self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
#         # Create the heatmap
#         sns.heatmap(self.pivot_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, center=0, ax=self.ax)
        
#         # Add "%" back into heatmap display
#         for text in self.ax.texts:
#             text.set_text(f"{float(text.get_text()):.2f}%")
            
#         # Add titles and labels
#         self.ax.set_title("Signal Performance Across Models")
#         self.ax.set_xlabel("Models")
#         self.ax.set_ylabel("Signals")
                
#         # Return the figure without displaying it
#         return self.fig
    
    
# #     def output_stats(self, output_folder='../Outputs'):
# #         #Saving heatmap as png to ../Outputs/202502041649/5
# #         plt.savefig(f"{output_folder}/signal_heatmap.png")
# #         plt.close()
    
# #         return f"Stats saved to {output_folder}"
    
# # # analyzer = SignalAnalyzer('../Outputs/202502041649/5/HeatmapData1.csv')
# # # analyzer.heatmap()
# # # analyzer.print_heatmap()
# #     ##png folder location
# #     def output_stats(self, output_folder='Outputs/202502041649/5'):
# #         self.fig.savefig(f"{output_folder}/signal_heatmap.png")

# # analyzer = Portfolio('../Outputs/202502041649/5/HeatmapData1.csv')
# # analyzer.heatmap()
# # analyzer.output_stats('../Outputs/202502041649/5')
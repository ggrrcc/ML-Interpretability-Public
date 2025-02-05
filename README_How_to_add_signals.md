## Steps

Research to find a signal. This can be something like accounting data, a ratio, whatever, as long as you can make a "story" out of it, as in reason why it would be a signal

Add the signal you want to add to the doc

Add to WRDS by looking through the fundamental or price databases using the saved queries- keep in mind that there are often duplicate variables/columns which may have a lot of NaN's. You'll have to check using testing_data_for_nans.ipynb

Now to add signals in the code:

- Add the signal to signal_info.py (2 places: top in the list, and in the function underneath. The format is signal_label_dict_og['YOUR_SIGNAL_HERE_Label'] = ['YOUR_SIGNAL_HERE']- note that if you have a more complicated signal you can have multiple entries in the value list)

- Add signals under "Signal Info" in TradingAlgorithms.ipynb. Write your story/reasoning here

Deleted Inputs/Created/merged.dta and merged_test.dta

Rerun first two cells of TradingAlgorithms.ipynb to recreate merged.dta or merged_test.dta
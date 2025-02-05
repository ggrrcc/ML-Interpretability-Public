# Machine Learning Interpretability
 Using 10-K filings and stock price information, we are aiming to build a trading algorithm that has higher interpretability than a normal machine-learning based trading algorithm.

 # SETUP
 Python version: 3.12.7
 
 ## 1 Install Python
 https://www.python.org/downloads/release/python-3127/
 
 ## 2 Install GitHub
 https://desktop.github.com/download/
 
 ### After installing remember to clone the repo

 ### Also remember to create a branch/switch branches
 
 ## 3 Install VS Code
 https://code.visualstudio.com/download
 
 Restart
 
 ## 4  Install Git
 https://git-scm.com/downloads
 
 Select what version of OS you have
 
 ### For Mac: 
 do homebrew method
 open terminal
 Command: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
 there's gonna be a thing that tells you how to add brew to path. Do that.
 Command: brew install git
 
 ### For Windows
 Remember to get winget
 
 ### Continue:
 Use Visual Studio Code as Git default editor
 
 Restart (if needed)
 
 ### IN VS CODE
 
 Open folder ML-Interpretability
 
 Terminal > New Terminal
 
 Optional: Change default terminal
 
 To create the virtual environment, use 
 python -m venv .venv (Mac may need to say python3 -m venv .venv)
 
 To activate the virtual env, use
 . .venv/Scripts/activate (Linux/MacOS? is . .venv/bin/activate)
 
 To upgrade pip:
 py -m ensurepip --upgrade (may need to say python3 -m ensurepip --upgrade)
 py -m pip install -U pip (or python3 -m pip install -U pip)
 
 To install all needed packages:
 pip install -r requirements.txt
 
 Try running (just the import statements). You'll get a popup saying to select python. click python environments and then the python in your .venv folder
 data processor
 backtest statistician
 Trading Algos
 
 pip install whatever you need

## Steps

Research to find a signal. This can be something like accounting data, a ratio, whatever, as long as you can make a "story" out of it, as in reason why it would be a signal

Add the signal you want to add to the doc

Add to WRDS by looking through the fundamental or price databases using the saved queries- keep in mind that there are often duplicate variables/columns which may have a lot of NaN's. You'll have to check using testing_data_for_nans.ipynb

Add the signal to idp_signals.py (2 places: top in the list, and under your name. the format is there)

Add the signal to defines.py in the format signal_label_dict_og['YOUR_SIGNAL_HERE_Label'] = ['YOUR_SIGNAL_HERE']- note that if you have a more complicated signal you can have multiple entries in the value list

Deleted Inputs/Created/merged.dta and merged_test.dta

Add signals under "Signal Info" in TradingAlgorithms.ipynb. Write your story/reasoning here

Rerun first two cells of TradingAlgorithms.ipynb to recreate merged.dta or merged_test.dta
# stock-market-project
=======
# Problem Description
- I want use the stock from alpha vantage but only free 25 ticker/day and paid quite expensive so i use choose yfinance api with many features

# Data Sources 
- I choose 25 stock from US and add several features. For the details see Data Source With New Features.ipynb file.
![alt text](image.png)
![alt text](image-1.png)

# Data Transformations + EDA
- Data is combined into one data frame. Feature sets are defined (TO_PREDICT, NUMERIC, DUMMIES)
- New relevant features are generated from transformations (at least 5. One dummy set is one feature): it can be binned variables from numeric features or manual transformations.
Create a new string column weekday and month columns
Create a new string column 'month_wom' (month_week_of_month)
Binned variables from numeric features
Interaction features
Lag features
- Data Analysis:
Correlation between df_dummies correlation with is_positive_growth_5d_future
corr_is_positive_growth_5d_future = df_with_dummies[NUMERICAL+DUMMIES+TO_PREDICT].corr()['is_positive_growth_5d_future']
Correlation between df_dummies correlation with growth future 5d
corr_growth_future_5d = df_with_dummies[NUMERICAL+DUMMIES+TO_PREDICT].corr()['growth_future_5d']
![alt text](image-3.png)
![alt text](image-4.png)
For the details see Data Transformations, EDA, Modeling, Trading Simulation.ipynb

# Modeling
I used DecisionTree algorithm, RandomForest algorithm, logistic regression algorithm and hyperparameter tuning for each algorithm. Then, custom decision rules on target higher probability events.
Also, i used deep neural networks and hyperparameter tuning in the end.

# Trading Simulation
- Vector simulations for at least 1 strategy (and approximate returns on capital).
- Two or more strategies are covered (sim1_, sim2_, etc. fields are generated for each prediction).
- Exact simulations (iter.rows) with reinvestment of capital gains and efficient capital utilization.
- Profitability discussion vs. benchmark, CAGR, Sharpe ratio, max drawdown, rolling returns, etc.
- The best strategy has advanced features: risk management (e.g., stop loss), time of entry/sell, increased investment with higher probability, portfolio optimization.
- New strategy: introduce a new empirical strategy based on the predictions, e.g., long-short strategy, or use no more than 1-3-5 concurrent investments, or combine with market conditions (trade only when volatility is high or current price is close to 52 weeks low), etc.
- Exceptional profitability: choose a realistic benchmark (e.g., S&P500 index) and show that your best prediction/strategy delivers better performance (CAGR) on average than a benchmark.
- Deep exploratory analysis: how predictions/precision are different by tickers (or markets, or month of year, or other features, etc.). Debug wrong predictions. Give ideas on the data features/models/strategies improvement based on the insights.

# Deployment and Automation Module

    Create a Pipfile with all dependencies.
    Install the dependencies using Pipenv.
    Generate requirements.txt from the Pipenv environment.

Step 1: Create a Pipfile

Create a Pipfile in your project directory with the following content:
[[source]]
name = "pypi"
url = "https://pypi.org/simple"
verify_ssl = true

[dev-packages]

[packages]
pandas = "*"
numpy = "*"
scikit-learn = "*"
joblib = "*"
tqdm = "*"
talib = "*"
yfinance = "*"
pandas-datareader = "*"

Step 2: Install the Dependencies

Run the following command to install the dependencies:
pipenv install

tep 3: Generate requirements.txt

After installing the dependencies, you can generate a requirements.txt file:
pipenv lock -r > requirements.txt

# Algo Trading

This project implements an end-to-end algorithmic trading strategy using the `yfinance` API.

## Installation

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment and install dependencies:

```sh
python -m venv env
source env/bin/activate   # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

# Local Automation Instructions

## Setting Up the Project Environment (in Terminal)

* Change the working directory to the Module5 folder: `cd 05-deployment-and-automation/`
* Install virtual environment: `pip3 install virtualenv`
* Create a new virtual environment (venv): `virtualenv venv` (or run `python3 -m venv venv`)
* Activate the new virtual environment: `source venv/bin/activate`

* Install all requirements to the new environment (venv): `pip3 install -r requirements.txt`

## Running the Project

* Start the local Jupyter Server (after activating venv): `jupyter notebook` (you can check all servers running with `jupyter notebook list`)
* Open `test.ipynb` to check the system's operation:
  * From your web browser (navigate to http://localhost:8888/tree or similar)
  * Or via the VS Code UI (specify the server address kernel) 
* Run `main.py` from the Terminal (or Cron) to simulate one new day of data.
>>>>>>> 3ce90e5 (first commit project)

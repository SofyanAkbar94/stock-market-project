<<<<<<< HEAD
# stock-market-project
=======
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

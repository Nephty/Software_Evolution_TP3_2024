import pandas as pd
from pandas import option_context
import numpy as np
import requests
import seaborn as sns
import datetime
import cliffsDelta as cd

from tqdm import tqdm
from statsmodels.stats.inter_rater import fleiss_kappa,aggregate_raters
from scipy.stats import mannwhitneyu,shapiro,ttest_ind

QUERY_ROOT = "https://api.github.com"
TOKEN = '' # write your GitHub API key here
HEADERS={'Authorization':'token '+TOKEN}

print("======")

# 1. Read the given .csv file into a pandas DataFrame (df).
# 2. Make sure there are 2 columns, one column name is "contributor" that has the name of the contributor, 
#    and another column name is "bothunter" that you will use later on in this assignment.
# 3. Display the DataFrame.

df = pd.read_csv("homebrew_homebrew-core_2/homebrew_homebrew-core_2.csv")

print(df.head())
print("\n"*3)
print(df.info())

print("======")

# 1. Store the names of the contributors in a list using the code snippet: 'df.contributor.to_list()'
# 2. print the list

contributors = df.contributor.to_list()
print(f"{contributors=}")

print("======")

# 1. Using a for loop, iterate over each contributor in the list and query the GitHub Users API.
#    You can use "query = f'{QUERY_ROOT}/users/{contributor}'", where QUERY_ROOT is defined at the beginning of this notebook 
#    above and 'contributor' is each individul contributor from the list
# 2. Get the response using 'response = requests.get(query, headers=HEADERS)'.
# 3. convert the response to JSON using 'json_response = response.json()'.
# 4. Iterate over this JSON response and get the value of the 'type' key. If it is "Bot" then the contributor is an App, 
#    if "User" then the contributor is an account. You should boolean values to indicate if the contributor is an App (True) or User/Organisation (False)
# 5. Save these results in list of dictionary of the form [{'contributor': <contributor name>, 'app': <boolean value>}, {...}, {...}, {...}].
#    Lets call this list as "app_list_dict"
# 6. Finally convert this list of dictionary to DataFrame by writing 'pd.DataFrame.from_dict(app_dict)'
# 7. Display the DataFrame. This should have two columns - contributor and app

app_list_dict = []

for contributor in contributors:
    query = f'{QUERY_ROOT}/users/{contributor}'
    response = requests.get(query, headers=HEADERS)
    json_response = response.json()
    print(json_response['type'])
    is_app = json_response['type'] == "Bot"
    username = json_response['login']
    app_list_dict.append({'contributor': username, 'app': is_app})

app_list_df = pd.DataFrame.from_dict(app_list_dict)

print(app_list_df)

print("======")

# Merge the app DataFrame to df by writing 'pd.merge(df, <app df>, on='contributor'). This is similar to SQL join on primary key 'contributor'.
# The resultant df should have 3 columns - contributor, bothunter and app.

merged = pd.merge(df, app_list_df, on='contributor')

print(merged)

print("======")

# Report on the total number of GitHub Apps, and User accounts present in the list of accounts of your dataset.

apps_count = len([dict for dict in app_list_dict if dict['app']])
users_count = len(app_list_dict) - apps_count
print(f"Total number of GitHub Apps : {apps_count}. Number of GitHub Users : {users_count}.")

print("======")

# Read the .csv file that has the predicitons given by BoDeGHa.

bodegha_pred = df = pd.read_csv("homebrew_homebrew-core_2/bodegha_predictions.csv")

print(bodegha_pred)

print("======")

# merge it to df - Now your df should have the following four columns - contributor, bothunter, app, bodegha

merged = df.merge(merged, how='left', left_on='account', right_on='contributor')
del merged['account']
merged.rename(columns={'prediction': 'bodegha'}, inplace=True)

print(merged)

print("======")

# Read the predictions given by RABBIT from the corresponding .csv file 

rabbit_pred = pd.read_csv("homebrew_homebrew-core_2/rabbit_predictions.csv")

print(rabbit_pred)

print("======")

# merge it to df - Now your df should have the following five columns - contributor, bothunter, app, bodegha, rabbit

merged = df.merge(merged, how='left', left_on='account', right_on='contributor')
del merged['account']
merged.rename(columns={'prediction': 'rabbit'}, inplace=True)

print(merged)

print("======")

# Replace the prediction result given by rabbit from 'bot' to 'Bot', 'app' to 'Bot', 'human' to 'Human', 'unknown' to 'Unknown' to maintain consistency

merged['rabbit'] = merged['rabbit'].replace({'bot': 'Bot', 'app': 'Bot', 'human': 'Human', 'unknown': 'Unknown'})
merged['bodegha'] = merged['bodegha'].replace({'bot': 'Bot', 'app': 'Bot', 'human': 'Human', 'unknown': 'Unknown'})

print(merged)

print("======")
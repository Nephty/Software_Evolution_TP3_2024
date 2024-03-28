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
del rabbit_pred['confidence']
del rabbit_pred['num']
rabbit_pred['prediction'] = rabbit_pred['prediction'].str.capitalize()

print(rabbit_pred)

print("======")

# merge it to df - Now your df should have the following five columns - contributor, bothunter, app, bodegha, rabbit

merged = pd.merge(merged, rabbit_pred, on='contributor')
#del merged['account']
merged.rename(columns={'prediction': 'rabbit'}, inplace=True)

print(merged)

print("======")

# Replace the prediction result given by rabbit from 'bot' to 'Bot', 'app' to 'Bot', 'human' to 'Human', 'unknown' to 'Unknown' to maintain consistency

merged['rabbit'] = merged['rabbit'].replace({'bot': 'Bot', 'app': 'Bot', 'human': 'Human', 'unknown': 'Unknown'})
merged['bodegha'] = merged['bodegha'].replace({'bot': 'Bot', 'app': 'Bot', 'human': 'Human', 'unknown': 'Unknown'})

print(merged)

print("======")

# Create a column named 'type' to the CSV ﬁle and write your ﬁnal decision on the type of contributor. The ﬁnal decision on their type is the prediciton that majority of the raters predicted (you can ignore the 'Unknown'), in the case of contradiction, you can finalise it as Human.

def determine_final_type(row):
    counts = row.value_counts()
    counts = counts.drop('Unknown', errors='ignore')

    # Case 1 : if two raters mentioned a contributor as Bot and others mentioned it as Unknown
    if 'Bot' in counts.index and counts['Bot'] >= 2:
        return 'Bot'
    
    # Case 2 : if 3 raters give Unknown to a contributor
    if counts.get('Unknown', 0) >= 3:
        return counts.idxmax()
    
    # Case 3 : if 2 raters give the prediction as Bot and 2 raters give it as Human
    if 'Bot' in counts.index and 'Human' in counts.index and counts['Bot'] == counts['Human']:
        return 'Human'
    
    # final decision is Human if there is contradiction
    return 'Human'

merged['type'] = merged[['rabbit', 'bodegha', 'bothunter']].apply(determine_final_type, axis=1)

# If they are an app we overwrite the decision to be App
merged.loc[merged['app'], 'type'] = 'App'

print(merged)

print("======")

# Compute and report the Fleiss Kappa interrater agreement score between the labels computed by all bot identiﬁcation tools.

# between bothunter and rabbit
converted_df = aggregate_raters(np.array(merged[['bothunter','rabbit']]))  
kappa = fleiss_kappa(converted_df[0])  
print(f"bothunter & rabbit : {kappa}")

# between bothunter and bodegha
converted_df = aggregate_raters(np.array(merged[['bothunter','bodegha']]))  
kappa = fleiss_kappa(converted_df[0])  
print(f"bothunter & bodegha : {kappa}")

# between bodegha and rabbit
converted_df = aggregate_raters(np.array(merged[['bodegha','rabbit']]))  
kappa = fleiss_kappa(converted_df[0])  
print(f"bodegha & rabbit : {kappa}")

# between bothunter, bodegha and rabbit
converted_df = aggregate_raters(np.array(merged[['bothunter','bodegha','rabbit']]))  
kappa = fleiss_kappa(converted_df[0])  
print(f"bothunter, bodegha & rabbit : {kappa}")

print("======")

# What is you interpretation from the kappa values (use the table provided in the description document)? what do you understand?

# Bothunter and rabbit agree on most decisions (0.83 > 0 and 0.83 is close to 1).
# Bothunter and bodegha also mostly agree but to a lesser extent (0.67).
# Bodegha and rabbit somewhat agree (0.52).
# When the three models come together, they agree as much as bothunter and bodegha do.
# All of this is only deducted from the current dataset of course.

# Based on your previous analysis, which bot identification tool would you use, if you want to predict if a contributor is a human or bot? Motivate your answer

# Probably bothunter and rabbit. This way, we do not rely on one single model and we have the highest kappa value.


all_events = []
for contributor in contributors:
    for i in range(1, 4):
        query = f'{QUERY_ROOT}/users/{contributor}/events?per_page=100&page={i}'
        response = requests.get(query, headers=HEADERS)
        json_response = response.json()
        print(f"Queried /users/{contributor}/events?per_page=100&page={i}")
        for event in json_response:
            event_details = {
                "event_id": event["id"],
                "time_of_event": event["created_at"],
                "type_of_event": event["type"],
                "repository_name": event["repo"]["name"],
                "contributor": event["actor"]["login"]
            }
            all_events.append(event_details)

_df = pd.DataFrame(all_events)
_df.to_csv("events.csv", index=False)

print("======")

# Group each event into the following four categories
# Issues: IssueCommentEvent, IssuesEvent  
# Pull Requests: PullRequestEvent, PullRequestReviewCommentEvent  
# Commits: CommitCommentEvent, PushEvent  
# Repository: CreateEvent, DeleteEvent, ForkEvent, GollumEvent, MemberEvent
#             PublicEvent, ReleaseEvent, SponsorshipEvent, WatchEvent  

issues_events = ["IssueCommentEvent", "IssuesEvent"]
pull_requests_events = ["PullRequestEvent", "PullRequestReviewCommentEvent"]
commits_events = ["CommitCommentEvent", "PushEvent"]
repository_events = ["CreateEvent", "DeleteEvent", "ForkEvent", "GollumEvent", "MemberEvent", 
                    "PublicEvent", "ReleaseEvent", "SponsorshipEvent", "WatchEvent"]

def categorize_event(event_type):
    if event_type in issues_events:
        return "Issues"
    elif event_type in pull_requests_events:
        return "Pull Requests"
    elif event_type in commits_events:
        return "Commits"
    elif event_type in repository_events:
        return "Repository"

events_df = pd.DataFrame(all_events)
events_df['event_group'] = events_df['type_of_event'].apply(categorize_event)
grouped_events = events_df.groupby(['contributor', 'event_group']).size().reset_index(name='event_count')
pivot_table = grouped_events.pivot(index='contributor', columns='event_group', values='event_count').fillna(0)
final_merged_df = pd.merge(pivot_table, merged[['contributor', 'type']], on='contributor', how='left')
final_merged_df['type'] = final_merged_df['type'].combine_first(rabbit_pred['prediction'])
print(final_merged_df)

print("======")

# Compute the median number of events per event group for Bot+Apps and Humans and write in DataFrame.
# Row should correspond to type (Bot_App and Human), Column should have Event group name and the values should be the median value of Bot_App or Human for that particular event group. An example is given below

bot_app_medians = final_merged_df[final_merged_df['type'] == 'App'][['Issues', 'Pull Requests', 'Commits', 'Repository']].median()
bot_app_medians.index.name = 'event_group'
bot_app_medians = bot_app_medians.reset_index()
bot_app_medians['type'] = 'Bot_App'
print(bot_app_medians)

human_medians = final_merged_df[final_merged_df['type'] == 'Human'][['Issues', 'Pull Requests', 'Commits', 'Repository']].median()
human_medians.index.name = 'event_group'
human_medians = human_medians.reset_index()
human_medians['type'] = 'Human'
print(human_medians)

all_medians = pd.concat([bot_app_medians, human_medians])
print(all_medians)

print("======")
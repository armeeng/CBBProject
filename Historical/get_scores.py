import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, inspect

pd.set_option('display.max_rows', None)  # So you can view the whole table
pd.set_option('display.max_columns', None)  # So you can view the whole table


# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

old_game_results_query = "SELECT * FROM game_results"
old_game_results = pd.read_sql(old_game_results_query, engine)

name_query = "SELECT * FROM team_names"
names = pd.read_sql(name_query, engine)

name_mapping = names.set_index('Team')['ESPNNames'].to_dict()
name_mapping = {v: k for k, v in name_mapping.items()}

# Send a GET request to the ESPN scoreboard page
url = "https://www.espn.com/mens-college-basketball/scoreboard/_/date/20230218/group/50"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")


columns = ["Team1", "Score1", "Team2", "Score2", "Status"]
df = pd.DataFrame(columns=columns)

# Extract all games
all_games = soup.find_all("li", class_="ScoreboardScoreCell__Item")
for i in range(0, len(all_games), 2):
    # Extract team names
    teams1 = all_games[i].find_all("div", class_="ScoreCell__TeamName")
    teams2 = all_games[i + 1].find_all("div", class_="ScoreCell__TeamName")

    team1 = teams1[0].text.strip() if teams1 and len(teams1) >= 1 else "N/A"
    team2 = teams2[0].text.strip() if teams2 and len(teams2) >= 1 else "N/A"

    # Extract scores
    scores1 = all_games[i].find_all("div", class_="ScoreCell__Score")
    scores2 = all_games[i + 1].find_all("div", class_="ScoreCell__Score")

    score1 = scores1[0].text.strip() if scores1 and len(scores1) >= 1 else "N/A"
    score2 = scores2[0].text.strip() if scores2 and len(scores2) >= 1 else "N/A"

    team1 = name_mapping.get(team1, team1)
    team2 = name_mapping.get(team2, team2)

    if (team1 in name_mapping and team2 in name_mapping) or (
            team1 in name_mapping.values() and team2 in name_mapping.values()):

        new_row = pd.DataFrame({"Team1": [team1], "Score1": [score1], "Team2": [team2], "Score2": [score2],
                                "Status": "Finished"})
        df = pd.concat([df, new_row], ignore_index=True)

for index, row in df.iterrows():
    team1 = row['Team1']
    team2 = row['Team2']

    # Check if the matchup is in old_game_results and has 'Complete' status
    if any(((old_game_results['Team1'] == team1) & (old_game_results['Team2'] == team2) |
            (old_game_results['Team1'] == team2) & (old_game_results['Team2'] == team1)) &
           (old_game_results['Status'] == 'Complete')):
        df.at[index, 'Status'] = 'Complete'

df.to_sql(name="game_results", con=engine, if_exists='replace', index=False)

print(df)


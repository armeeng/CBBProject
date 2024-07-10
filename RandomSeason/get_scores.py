import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
import datetime

pd.set_option('display.max_rows', None)  # So you can view the whole table
pd.set_option('display.max_columns', None)  # So you can view the whole table

# MySQL Database Connection Details
####removed

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
connection_str_historical = f"mysql+pymysql://{db_username_historical}:{db_password_historical}@{db_host_historical}:{db_port_historical}/{db_name_historical}"


# Create SQLAlchemy Engine
engine = create_engine(connection_str)
engine_historical = create_engine(connection_str_historical)


name_query = "SELECT * FROM team_names"
names = pd.read_sql(name_query, engine_historical)

name_mapping = names.set_index('Team')['ESPNNames'].to_dict()
name_mapping = {v: k for k, v in name_mapping.items()}

# Function to scrape and insert data for a given date
def scrape_and_insert(date):
    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
    date2 = date_obj.strftime("%Y%m%d")

    # Send a GET request to the ESPN scoreboard page
    url = f"https://www.espn.com/mens-college-basketball/scoreboard/_/date/{date2}/group/50"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    columns = ["Team1", "Score1", "Team2", "Score2", "Date", "Status"]
    df = pd.DataFrame(columns=columns)

    # Extract all games
    all_games = soup.find_all("li", class_="ScoreboardScoreCell__Item")
    for i in range(0, len(all_games), 2):
        # Extract team names
        teams1 = all_games[i].find_all("div", class_="ScoreCell__TeamName")
        teams2 = all_games[i + 1].find_all("div", class_="ScoreCell__TeamName")

        team1 = teams1[0].text.strip() if teams1 and len(teams1) >= 1 else "N/A"
        team2 = teams2[0].text.strip() if teams2 and len(teams2) >= 1 else "N/A"

        # Check if both team1 and team2 are in the name_mapping
        if team1 in name_mapping and team2 in name_mapping:
            # Extract scores
            scores1 = all_games[i].find_all("div", class_="ScoreCell__Score")
            scores2 = all_games[i + 1].find_all("div", class_="ScoreCell__Score")

            score1 = scores1[0].text.strip() if scores1 and len(scores1) >= 1 else "N/A"
            score2 = scores2[0].text.strip() if scores2 and len(scores2) >= 1 else "N/A"

            # Add the game to the DataFrame
            new_row = pd.DataFrame({"Team1": [name_mapping.get(team1, team1)], "Score1": [score1],
                                     "Team2": [name_mapping.get(team2, team2)], "Score2": [score2],
                                     "Date": [date], "Status": "Finished"})
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_sql(name="game_results", con=engine, if_exists='append', index=False)

# Loop through dates from January 1st to March 1st
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 3, 1)
delta = datetime.timedelta(days=1)
current_date = start_date

while current_date <= end_date:
    scrape_and_insert(current_date.strftime("%Y-%m-%d"))
    current_date += delta
    print(current_date)

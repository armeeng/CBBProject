import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, inspect
from datetime import datetime


pd.set_option('display.max_rows', None)  # So you can view the whole table
pd.set_option('display.max_columns', None)  # So you can view the whole table

def increment_string(input_str, num):
    if num not in {1, 2}:
        raise ValueError("Invalid value for num. Use 1 for updating W or 2 for updating L.")

    # Split the input string to extract W, L, and %
    parts = input_str.split(', ')
    w_value = int(parts[0].split(': ')[1])
    l_value = int(parts[1].split(': ')[1])

    if num == 1:
        # Increment the value associated with W
        w_value += 1
    elif num == 2:
        # Increment the value associated with L
        l_value += 1

    # Calculate the new W/L percentage
    total_games = w_value + l_value
    win_percentage = (w_value / total_games) if total_games > 0 else 0

    # Check if there are more losses than wins
    if l_value > w_value:
        win_percentage = -win_percentage

    # Update the string with the new values
    updated_str = f"W: {w_value}, L: {l_value}, %: {win_percentage:.2f}"

    return updated_str

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

today_date = datetime.today().strftime('%Y-%m-%d')

# Load the schedule table
schedule_query = "SELECT * FROM Schedule"
schedule_table_data = pd.read_sql(schedule_query, engine)

summary_query = "SELECT * FROM summary"
summary_table_data = pd.read_sql(summary_query, engine)

old_game_results_query = "SELECT * FROM game_results"
old_game_results = pd.read_sql(old_game_results_query, engine)

record_query = "SELECT * FROM record"
record = pd.read_sql(record_query, engine)
old_record = pd.read_sql(record_query, engine)
old_record.to_sql(name="old_record", con=engine, if_exists='replace', index=False)

record= record.sort_values(by='Time', ascending=False)


# Send a GET request to the ESPN scoreboard page
url = "https://www.teamrankings.com/ncb/scores/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

columns = ["Team1", "Score1", "Team2", "Score2", "Status", "Winner", "DataDifferencesSum", "MultDataDifferencesSum", "MultDataDifferencesHistoricalSum", "RandomSum", "RandomTournamentSum", 'FNNSeason', 'FNNMoreSeason', 'FNNTournament', 'FNNMoreTournament', "Odds", "Record Status"]
df = pd.DataFrame(columns=columns)

record_done = any(record['Time'] == today_date)

if record_done:
    existing_record_row = record[record['Time'] == today_date].iloc[0]

    new_record_row = {
        "Time": existing_record_row["Time"],
        "DataDifferencesSum": existing_record_row["DataDifferencesSum"],
        "MultDataDifferencesSum": existing_record_row["MultDataDifferencesSum"],
        "MultDataDifferencesHistoricalSum": existing_record_row["MultDataDifferencesHistoricalSum"],
        "RandomSum": existing_record_row["RandomSum"],
        "RandomTournamentSum": existing_record_row["RandomTournamentSum"],
        "FNNSeason": existing_record_row["FNNSeason"],
        "FNNMoreSeason": existing_record_row["FNNMoreSeason"],
        "FNNTournament": existing_record_row["FNNTournament"],
        "FNNMoreTournament": existing_record_row["FNNMoreTournament"],
        "Odds": existing_record_row["Odds"]
    }
else:
    new_record_row = {
        "Time": today_date,
        "DataDifferencesSum": "W: 0, L: 0, %: 0",
        "MultDataDifferencesSum": "W: 0, L: 0, %: 0",
        "MultDataDifferencesHistoricalSum": "W: 0, L: 0, %: 0",
        "RandomSum": "W: 0, L: 0, %: 0",
        "RandomTournamentSum": "W: 0, L: 0, %: 0",
        "FNNSeason": "W: 0, L: 0, %: 0",
        "FNNMoreSeason": "W: 0, L: 0, %: 0",
        "FNNTournament": "W: 0, L: 0, %: 0",
        "FNNMoreTournament": "W: 0, L: 0, %: 0",
        "Odds": "W: 0, L: 0, %: 0"
    }

# Extract all games
all_games = soup.find_all("table", class_="tr-table")
for game in all_games:
    header = game.find("th").text.strip()
    is_finished = "FINAL" in header

    teams = game.select("td strong a")
    team1 = teams[0].text if teams and len(teams) >= 1 else "N/A"
    team2 = teams[1].text if teams and len(teams) >= 2 else "N/A"

    for index, row in summary_table_data.iterrows():
        if (row['Team1'] == team1) & (row['Team2'] == team2):
            data_row = row

    if 'data_row' in locals():
        data_differences_sum = team2 if data_row["DataDifferencesSum"] < 0 else team1
        mult_data_differences_sum = team2 if data_row["MultDataDifferencesSum"] < 0 else team1
        mult_data_differences_historical_sum = team2 if data_row["MultDataDifferencesHistoricalSum"] < 0 else team1
        random_sum = team2 if data_row["Random"] < 0 else team1
        tournament_sum = team2 if data_row["RandomTournament"] < 0 else team1
        fnnSeason_sum = team2 if data_row["FNNSeason"] < 0 else team1
        fnnMoreSeason_sum = team2 if data_row["FNNMoreSeason"] < 0 else team1
        fnnTournament_sum = team2 if data_row["FNNTournament"] < 0 else team1
        fnnMoreTournament_sum = team2 if data_row["FNNMoreTournament"] < 0 else team1

        if data_row["Odds"] < 0:
            odds = team2
        elif data_row["Odds"] > 0:
            odds = team1
        elif data_row["Odds"] == 0:
            odds = "None"

    # Check if the matchup is in the schedule table
    if any((schedule_table_data['Team1'] == team1) & (schedule_table_data['Team2'] == team2) |
           (schedule_table_data['Team1'] == team2) & (schedule_table_data['Team2'] == team1)):
        if is_finished:
            scores = game.select("td.points")
            score1 = scores[0].text if scores and len(scores) >= 1 else "N/A"
            score2 = scores[1].text if scores and len(scores) >= 2 else "N/A"
            status = "Finished"
            winner = team1 if int(score1) > int(score2) else team2

        else:
            score1 = "N/A"
            score2 = "N/A"
            status = "Not Finished"
            winner = "None"

        record_status = "Not Finished"

        new_row = pd.DataFrame(
            {"Team1": [team1], "Score1": [score1], "Team2": [team2], "Score2": [score2], "Status": [status], "Winner": winner, "DataDifferencesSum": data_differences_sum, "MultDataDifferencesSum": mult_data_differences_sum, "MultDataDifferencesHistoricalSum": mult_data_differences_historical_sum, "RandomSum": random_sum, "RandomTournamentSum": tournament_sum, 'FNNSeason': fnnSeason_sum, 'FNNMoreSeason': fnnMoreSeason_sum, 'FNNTournament': fnnTournament_sum, 'FNNMoreTournament': fnnMoreTournament_sum, "Odds": odds, "Record Status": record_status})
        df = pd.concat([df, new_row], ignore_index=True)

for index, row in df.iterrows():
    team1 = row['Team1']
    team2 = row['Team2']

    DataDiff = False
    MultDiff = False
    HistDiff = False
    OddsDiff = False
    RandDiff = False
    TournDiff = False
    fnnS = False
    fnnMS = False
    fnnT = False
    fnnMT = False

    # Check if the matchup is in old_game_results and has 'Complete' status
    if any(((old_game_results['Team1'] == team1) & (old_game_results['Team2'] == team2) |
            (old_game_results['Team1'] == team2) & (old_game_results['Team2'] == team1)) &
           (old_game_results['Status'] == 'Complete')):
        df.at[index, 'Status'] = 'Complete'

    if any(((old_game_results['Team1'] == team1) & (old_game_results['Team2'] == team2) |
            (old_game_results['Team1'] == team2) & (old_game_results['Team2'] == team1)) &
           (old_game_results['Record Status'] == 'Complete')):
        df.at[index, 'Record Status'] = 'Complete'

    winner = row['Winner']

    if row['DataDifferencesSum'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'DataDifferencesSum'] = increment_string(record.at[0, 'DataDifferencesSum'], 1)
        new_record_row['DataDifferencesSum'] = increment_string(new_record_row['DataDifferencesSum'], 1)
        DataDiff = True
    elif row['DataDifferencesSum'] != winner and (winner != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'DataDifferencesSum'] = increment_string(record.at[0, 'DataDifferencesSum'], 2)
        new_record_row['DataDifferencesSum'] = increment_string(new_record_row['DataDifferencesSum'], 2)
        DataDiff = True
    if row['MultDataDifferencesSum'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'MultDataDifferencesSum'] = increment_string(record.at[0, 'MultDataDifferencesSum'], 1)
        new_record_row['MultDataDifferencesSum'] = increment_string(new_record_row['MultDataDifferencesSum'], 1)
        MultDiff = True
    elif row['MultDataDifferencesSum'] != winner and (winner != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'MultDataDifferencesSum'] = increment_string(record.at[0, 'MultDataDifferencesSum'], 2)
        new_record_row['MultDataDifferencesSum'] = increment_string(new_record_row['MultDataDifferencesSum'], 2)
        MultDiff = True
    if row['MultDataDifferencesHistoricalSum'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'MultDataDifferencesHistoricalSum'] = increment_string(
            record.at[0, 'MultDataDifferencesHistoricalSum'], 1)
        new_record_row['MultDataDifferencesHistoricalSum'] = increment_string(
            new_record_row['MultDataDifferencesHistoricalSum'], 1)
        HistDiff = True
    elif row['MultDataDifferencesHistoricalSum'] != winner and (winner != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'MultDataDifferencesHistoricalSum'] = increment_string(
            record.at[0, 'MultDataDifferencesHistoricalSum'], 2)
        new_record_row['MultDataDifferencesHistoricalSum'] = increment_string(
            new_record_row['MultDataDifferencesHistoricalSum'], 2)
        HistDiff = True
    if row['RandomSum'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'RandomSum'] = increment_string(record.at[0, 'RandomSum'], 1)
        new_record_row['RandomSum'] = increment_string(new_record_row['RandomSum'], 1)
        RandDiff = True
    elif row['RandomSum'] != winner and (winner != "None") and (row['RandomSum'] != "None") and (
        row["Record Status"] != "Complete"):
        record.at[0, 'RandomSum'] = increment_string(record.at[0, 'RandomSum'], 2)
        new_record_row['RandomSum'] = increment_string(new_record_row['RandomSum'], 2)
        RandDiff = True
    if row['RandomTournamentSum'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'RandomTournamentSum'] = increment_string(record.at[0, 'RandomTournamentSum'], 1)
        new_record_row['RandomTournamentSum'] = increment_string(new_record_row['RandomTournamentSum'], 1)
        TournDiff = True
    elif row['RandomTournamentSum'] != winner and (winner != "None") and (row['RandomTournamentSum'] != "None") and (
        row["Record Status"] != "Complete"):
        record.at[0, 'RandomTournamentSum'] = increment_string(record.at[0, 'RandomTournamentSum'], 2)
        new_record_row['RandomTournamentSum'] = increment_string(new_record_row['RandomTournamentSum'], 2)
        TournDiff = True
    if row['Odds'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'Odds'] = increment_string(record.at[0, 'Odds'], 1)
        new_record_row['Odds'] = increment_string(new_record_row['Odds'], 1)
        OddsDiff = True
    elif row['Odds'] != winner and (winner != "None") and (row['Odds'] != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'Odds'] = increment_string(record.at[0, 'Odds'], 2)
        new_record_row['Odds'] = increment_string(new_record_row['Odds'], 2)
        OddsDiff = True
    if row['FNNSeason'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNSeason'] = increment_string(record.at[0, 'FNNSeason'], 1)
        new_record_row['FNNSeason'] = increment_string(new_record_row['FNNSeason'], 1)
        fnnS = True
    elif row['FNNSeason'] != winner and (winner != "None") and (row['FNNSeason'] != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNSeason'] = increment_string(record.at[0, 'FNNSeason'], 2)
        new_record_row['FNNSeason'] = increment_string(new_record_row['FNNSeason'], 2)
        fnnS = True
    if row['FNNMoreSeason'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNMoreSeason'] = increment_string(record.at[0, 'FNNMoreSeason'], 1)
        new_record_row['FNNMoreSeason'] = increment_string(new_record_row['FNNMoreSeason'], 1)
        fnnMS = True
    elif row['FNNMoreSeason'] != winner and (winner != "None") and (row['FNNMoreSeason'] != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNMoreSeason'] = increment_string(record.at[0, 'FNNMoreSeason'], 2)
        new_record_row['FNNMoreSeason'] = increment_string(new_record_row['FNNMoreSeason'], 2)
        fnnMS = True
    if row['FNNTournament'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNTournament'] = increment_string(record.at[0, 'FNNTournament'], 1)
        new_record_row['FNNTournament'] = increment_string(new_record_row['FNNTournament'], 1)
        fnnT = True
    elif row['FNNTournament'] != winner and (winner != "None") and (row['FNNTournament'] != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNTournament'] = increment_string(record.at[0, 'FNNTournament'], 2)
        new_record_row['FNNTournament'] = increment_string(new_record_row['FNNTournament'], 2)
        fnnT = True
    if row['FNNMoreTournament'] == winner and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNMoreTournament'] = increment_string(record.at[0, 'FNNMoreTournament'], 1)
        new_record_row['FNNMoreTournament'] = increment_string(new_record_row['FNNMoreTournament'], 1)
        fnnMT = True
    elif row['FNNMoreTournament'] != winner and (winner != "None") and (row['FNNMoreTournament'] != "None") and (row["Record Status"] != "Complete"):
        record.at[0, 'FNNMoreTournament'] = increment_string(record.at[0, 'FNNMoreTournament'], 2)
        new_record_row['FNNMoreTournament'] = increment_string(new_record_row['FNNMoreTournament'], 2)
        fnnMT = True

    if DataDiff and MultDiff and HistDiff and RandDiff and TournDiff and fnnS and fnnMS and fnnT and fnnMT:

        df.at[index, 'Record Status'] = 'Complete'

if record_done:
    # Update the existing record row
    existing_index = record[record['Time'] == today_date].index[0]
    record.at[existing_index, 'DataDifferencesSum'] = new_record_row['DataDifferencesSum']
    record.at[existing_index, 'MultDataDifferencesSum'] = new_record_row['MultDataDifferencesSum']
    record.at[existing_index, 'MultDataDifferencesHistoricalSum'] = new_record_row['MultDataDifferencesHistoricalSum']
    record.at[existing_index, 'RandomSum'] = new_record_row['RandomSum']
    record.at[existing_index, 'RandomTournamentSum'] = new_record_row['RandomTournamentSum']
    record.at[existing_index, 'FNNSeason'] = new_record_row['FNNSeason']
    record.at[existing_index, 'FNNMoreSeason'] = new_record_row['FNNMoreSeason']
    record.at[existing_index, 'FNNTournament'] = new_record_row['FNNTournament']
    record.at[existing_index, 'FNNMoreTournament'] = new_record_row['FNNMoreTournament']
    record.at[existing_index, 'Odds'] = new_record_row['Odds']
else:
    # Append the new record row
    record = pd.concat([record, pd.DataFrame([new_record_row])], ignore_index=True)

record.to_sql(name="record", con=engine, if_exists='replace', index=False)
df.to_sql(name="game_results", con=engine, if_exists='replace', index=False)
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO
import os
from sklearn.preprocessing import MinMaxScaler
import re
from selenium import webdriver

startTime = datetime.now()

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

scaler = MinMaxScaler()

url_list1 = [
    "https://www.teamrankings.com/ncaa-basketball/stat/points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/average-scoring-margin",
    "https://www.teamrankings.com/ncaa-basketball/stat/offensive-efficiency",
    "https://www.teamrankings.com/ncaa-basketball/stat/floor-percentage",
    "https://www.teamrankings.com/ncaa-basketball/stat/1st-half-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/2nd-half-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/overtime-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/average-1st-half-margin",
    "https://www.teamrankings.com/ncaa-basketball/stat/average-2nd-half-margin",
    "https://www.teamrankings.com/ncaa-basketball/stat/average-overtime-margin",
    "https://www.teamrankings.com/ncaa-basketball/stat/points-from-2-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/points-from-3-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-2-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-3-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-free-throws",
    "https://www.teamrankings.com/ncaa-basketball/stat/shooting-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/effective-field-goal-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/three-point-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/two-point-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/free-throw-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/true-shooting-percentage",
    "https://www.teamrankings.com/ncaa-basketball/stat/field-goals-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/field-goals-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/three-pointers-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/three-pointers-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/free-throws-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/free-throws-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/three-point-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/two-point-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/fta-per-fga",
    "https://www.teamrankings.com/ncaa-basketball/stat/ftm-per-100-possessions",
    "https://www.teamrankings.com/ncaa-basketball/stat/free-throw-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/non-blocked-2-pt-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/offensive-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/defensive-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/team-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/total-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/offensive-rebounding-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/defensive-rebounding-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/total-rebounding-percentage",
    "https://www.teamrankings.com/ncaa-basketball/stat/blocks-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/steals-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/block-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/steals-perpossession",
    "https://www.teamrankings.com/ncaa-basketball/stat/steal-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/turnovers-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/turnovers-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/assist--per--turnover-ratio",
    "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-fgm",
    "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/turnover-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/personal-fouls-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/personal-fouls-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/personal-foul-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-average-scoring-margin",
    "https://www.teamrankings.com/ncaa-basketball/stat/defensive-efficiency",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-floor-percentage",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-1st-half-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-2nd-half-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-overtime-points-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-from-2-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-from-3-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-2-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-3-pointers",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-free-throws",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-shooting-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-effective-field-goal-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-point-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-two-point-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throw-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-true-shooting-percentage",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-field-goals-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-field-goals-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-pointers-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-pointers-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throws-made-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throws-attempted-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-point-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-two-point-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-fta-per-fga",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-ftm-per-100-possessions",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throw-rate",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-non-blocked-2-pt-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-offensive-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-defensive-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-team-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-total-rebounds-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-offensive-rebounding-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-defensive-rebounding-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-blocks-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steals-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-block-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steals-perpossession",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steal-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnovers-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assist--per--turnover-ratio",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-fgm",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnovers-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnover-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-fouls-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-fouls-per-possession",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-foul-pct",
    "https://www.teamrankings.com/ncaa-basketball/stat/possessions-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/extra-chances-per-game",
    "https://www.teamrankings.com/ncaa-basketball/stat/effective-possession-ratio",
    "https://www.teamrankings.com/ncaa-basketball/stat/opponent-effective-possession-ratio", #1
]

url_list2 = [
    "https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/home-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/away-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/neutral-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/home-adv-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/luck-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/consistency-by-other",
]

url_list3 = [
    "https://www.teamrankings.com/ncaa-basketball/ranking/schedule-strength-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/future-sos-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/season-sos-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/sos-basic-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/in-conference-sos-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/non-conference-sos-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/last-5-games-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/last-10-games-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/in-conference-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/non-conference-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/vs-1-25-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/vs-26-50-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/vs-51-100-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/vs-101-200-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/vs-201-and-up-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/first-half-by-other",
    "https://www.teamrankings.com/ncaa-basketball/ranking/second-half-by-other",
]

def remove_columns_by_names(df, *column_names):
    """
    Remove columns with exact specified names from a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - *column_names: variable arguments for exact column names to be removed

    Returns:
    - Modified DataFrame
    """
    df = df.drop(columns=[col for col in df.columns if col in column_names], errors='ignore')
    return df

def remove_symbols(df):
    """
    Remove '%' symbol from all cells in the DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - Modified DataFrame
    """
    # Apply replace to the entire DataFrame
    df = df.replace('%', '', regex=True)

    return df

def convert_ratio_to_percentage(df):
    """
    Convert ratio-like records (e.g., '2-1') to percentages in the DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - Modified DataFrame
    """
    # Define the ratio_to_percentage function
    def ratio_to_percentage(ratio):
        parts = [int(part) if part.isdigit() else part for part in ratio.split('-')]
        if all(isinstance(part, int) for part in parts):
            return parts[0] / sum(parts) if sum(parts) != 0 else 0  # Avoid division by zero
        else:
            return ratio  # Leave non-numeric values unchanged

    # Identify columns with ratios
    ratio_columns = df.columns[df.apply(lambda x: '-' in str(x.iloc[0]), axis=0)]

    # Convert ratio-like records to percentages for all rows in ratio columns
    df[ratio_columns] = df[ratio_columns].apply(lambda x: x.apply(lambda y: ratio_to_percentage(y) if isinstance(y, str) else y))

    return df

def clean_team_names(df):
    """
    Remove records next to team names in the second column of the DataFrame, except for cases with '-'.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - Modified DataFrame
    """
    first_column = df.columns[0]
    # Assuming the second column contains team names
    df[first_column] = df[first_column].str.replace(r'\((?!FL|OH|NY|PA\)).*\)', '', regex=True)

    df[first_column] = df[first_column].str.strip()

    return df

def convert_to_numeric(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except (ValueError, TypeError):
            # Ignore columns that cannot be converted to numeric
            pass
    return df

def multiply_values_by_minus_one_if_ascending(df, numeric_columns):
    """
    Multiply all numeric values in the DataFrame by -1 if the second column is in ascending order.
    Exclude specific columns from multiplication.

    Parameters:
    - df: pandas DataFrame
    - numeric_columns: list of columns with numeric values

    Returns:
    - Modified DataFrame
    """

    # Define columns to be excluded from multiplication
    excluded_columns = ['v 1-25', 'v 26-50', 'v 51-100']

    # Check if the second column is in ascending order
    is_ascending = df.iloc[0, 1] <= df.iloc[-1, 1]

    if is_ascending:
        # Exclude specified columns from multiplication
        columns_to_multiply = [col for col in numeric_columns if col not in excluded_columns]
        df[columns_to_multiply] = df[columns_to_multiply] * -1

    return df

def process_data_to_sql (url_list, name):
    for url in url_list:
        # For loop used to go through each of the url's in the list

        WebContentsNorm = requests.get(url)
        # Gets the content from the website

        ReadContentsNorm = BeautifulSoup(WebContentsNorm.content, "lxml")
        # Uses Beautiful Soup to parse through the url contents

        FindTableNorm = ReadContentsNorm.find("table")
        # Uses Beautiful Soup to go through the website content and find "table"
        # Do something with the data here, such as storing it in a list or a DataFrame

        PresentTableNorm = pd.read_html(StringIO(str(FindTableNorm)), flavor="lxml")[0]

        PresentTableNorm = remove_columns_by_names(PresentTableNorm, 'Rank', 'Hi', 'Low', 'Last')

        PresentTableNorm = remove_symbols(PresentTableNorm)

        PresentTableNorm = convert_ratio_to_percentage(PresentTableNorm)

        PresentTableNorm = clean_team_names(PresentTableNorm)

        PresentTableNorm = PresentTableNorm.replace("--", 0, regex=True)

        PresentTableNorm = convert_to_numeric(PresentTableNorm)

        # Select only numeric columns
        numeric_columns = PresentTableNorm.select_dtypes(include=['number']).columns

        # Apply the scaler to numeric columns
        PresentTableNorm[numeric_columns] = scaler.fit_transform(PresentTableNorm[numeric_columns])

        PresentTableNorm = multiply_values_by_minus_one_if_ascending(PresentTableNorm, numeric_columns)

        # Insert DataFrame into MySQL
        PresentTableNorm.to_sql(name=name + os.path.basename(url), con=engine, if_exists="replace", index=False)

process_data_to_sql(url_list1, "url1_")
process_data_to_sql(url_list2, "url2_")
process_data_to_sql(url_list3, "url3_")

# Get today's date and format it as 'YYYY-MM-DD'
todays_date = datetime.now().strftime('%Y-%m-%d')

# Update the Schedule URL with today's date
Schedule = f"https://www.teamrankings.com/ncb/schedules/?date={todays_date}"

WebContentsSchedule = requests.get(Schedule)

ReadContentsSchedule = BeautifulSoup(WebContentsSchedule.content, "lxml")

FindTableSchedule = ReadContentsSchedule.find("table")

PresentTableSchedule = pd.read_html(StringIO(str(FindTableSchedule)), flavor="lxml")[0]

Mask = PresentTableSchedule['Hotness Score'] != '--'
PresentTableSchedule = PresentTableSchedule.drop(PresentTableSchedule[~Mask].index)
regex = r"#\d+"
PresentTableSchedule['Matchup'] = PresentTableSchedule['Matchup'].apply(lambda x: re.sub(regex, '', x))
PresentTableSchedule['Matchup'] = PresentTableSchedule['Matchup'].str.strip()

# Split 'Matchup' column into 'Team1' and 'Team2'
teams = PresentTableSchedule['Matchup'].str.split(re.compile(r'\s+at\s+|\s+vs\.\s+'))

# Create a DataFrame from the split data
teams_df = pd.DataFrame(teams.tolist(), columns=['Team1', 'Team2'])

odds_url = "https://www.teamrankings.com/ncb/odds/"

driver = webdriver.Chrome()
driver.get(odds_url)

# Wait for the page to load (adjust the time as needed)
driver.implicitly_wait(10)

# Find the div with class "module-in clear"
module_div = driver.find_element("css selector", 'div.module-in.clear')

# Find all tables within the div
tables = module_div.find_elements("tag name", 'table')

money_line_odds_dict = {}

# Loop through each table and extract HTML content
for table in tables:
    # Get the HTML content of the table
    html_content = table.get_attribute('outerHTML')

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the Money Line odds from the table
    money_line_tags = soup.select('tbody tr td:nth-child(5)')

    # Extract odds values from the tags
    money_line_odds = [tag.get_text(strip=True) for tag in money_line_tags]

    # Extract team names from the table
    team_names = soup.select('tbody tr td:nth-child(1) a')

    # Extract team names from the tags
    team_names = [name.get_text(strip=True) for name in team_names]

    # Map Money Line odds to team names in the dictionary
    for team, odds in zip(team_names, money_line_odds):
        money_line_odds_dict[team] = odds

# Map Money Line odds from the dictionary to the 'Money Line Odds' column in teams_df
teams_df['Money Line Odds'] = teams_df['Team2'].map(money_line_odds_dict)

# Close the webdriver
driver.quit()

# Save the DataFrame to MySQL
teams_df.to_sql(name="schedule", con=engine, if_exists="replace", index=False)

# Display the resulting DataFrame
print(teams_df)

endTime = datetime.now()
print(endTime - startTime)



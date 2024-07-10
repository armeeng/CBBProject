import subprocess
from datetime import datetime
import configparser
from sqlalchemy import create_engine
import pandas as pd

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

# Function to update the date in the config.ini file
def update_config_date(new_date):
    config = configparser.ConfigParser()
    config.read('config.ini')
    config.set('general', 'date', new_date)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Query to fetch unique dates from the game_results table
unique_dates_query = "SELECT DISTINCT Date FROM game_results"
unique_dates = pd.read_sql(unique_dates_query, engine)

# Iterate over each unique date
for index, row in unique_dates.iterrows():
    current_date_str = datetime.strptime(row['Date'], '%Y-%m-%d').strftime('%Y-%m-%d')
    print(current_date_str)

    # Update the date in config.ini
    update_config_date(current_date_str)

    # Run script1
    subprocess.run(["python", "scrape_data.py"])

    # Run script2
    subprocess.run(["python", "process_data_differences.py"])

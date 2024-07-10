import subprocess
from datetime import datetime, timedelta
import configparser

# Function to update the date in the config.ini file
def update_config_date(new_date):
    config = configparser.ConfigParser()
    config.read('config.ini')
    config.set('general', 'date', new_date)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Function to get the current date from the config.ini file
def get_current_date():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['general']['date']

# Define start and end dates
start_date_str = '2023-01-01'
end_date_str = '2023-03-01'

current_date_str = start_date_str
current_date = datetime.strptime(current_date_str, '%Y-%m-%d')

# Loop until the current date reaches the end date
while current_date_str <= end_date_str:
    # Run script1
    subprocess.run(["python", "scrape_data.py"])

    # Run script2
    subprocess.run(["python", "process_data_differences.py"])

    # Update the date in config.ini by adding one day
    current_date += timedelta(days=1)
    current_date_str = current_date.strftime('%Y-%m-%d')
    update_config_date(current_date_str)

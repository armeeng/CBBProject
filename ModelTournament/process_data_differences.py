from sqlalchemy import create_engine, inspect
import pandas as pd
import configparser


pd.set_option('display.max_rows', None) # So you can view the whole table
pd.set_option('display.max_columns', None) # So you can view the whole table

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

game_results_query = "SELECT * FROM game_results"
game_results_table_data = pd.read_sql(game_results_query, engine)

naming_patterns = ["url1_", "url2_", "url3_"]
summary_data = pd.DataFrame(columns=['Team1', 'Team2', 'DataDifferencesSum', 'MultDataDifferencesSum', 'MultDataDifferencesHistoricalSum', 'Odds'])

def pull_data_from_sql(naming_pattern, condition_column, condition_value):
    # Use the inspector to get the table names
    inspector = inspect(engine)
    table_names = [table_name for table_name in inspector.get_table_names() if naming_pattern in table_name]

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate through the matching tables and fetch data
    for table_name in table_names:
        # Remove the naming pattern from the table name
        clean_table_name = table_name.replace(naming_pattern, '')

        # Properly escape the table name
        escaped_table_name = f'`{table_name}`'

        # Build the SQL query with the escaped table name and condition
        query = f"SELECT * FROM {escaped_table_name} WHERE {condition_column} = '{condition_value}'"
        table_data = pd.read_sql(query, engine)

        # Add a new column "Table" with the clean table name for each row
        table_data['Stat'] = clean_table_name

        # Concatenate the table data to the combined data
        combined_data = pd.concat([combined_data, table_data], ignore_index=True)

    return combined_data

def subtract_tables(data_table1, data_table2):
    # Assuming both tables have the same structure
    # Subtract corresponding values from table2 to table1
    numeric_columns = data_table1.select_dtypes(include=['number']).columns
    result_table = data_table1[numeric_columns].subtract(data_table2[numeric_columns])

    # Add the 'Table' column from data_table1 to the result_table
    result_table['Stat'] = data_table1['Stat']

    return result_table


# Read date from config file
config = configparser.ConfigParser()
config.read('config.ini')
date_of_data = config['general']['date']

for index, row in game_results_table_data.iterrows():
    Team1 = row.iloc[0]
    Team2 = row.iloc[2]
    date_of_game = row.iloc[4]

    # Check if the date of the game matches the specified date_of_data
    if date_of_game == date_of_data:
        team1_row_indices = game_results_table_data.loc[(game_results_table_data['Team1'] == Team1)].index
        team2_row_indices = game_results_table_data.loc[(game_results_table_data['Team2'] == Team2)].index
        date_indices = game_results_table_data.loc[(game_results_table_data['Date'] == date_of_game)].index

        common_indices = set(team1_row_indices) & set(team2_row_indices) & set(date_indices)
        j = 1
        print(date_of_game)
        for naming_pattern in naming_patterns:
            team1_data = pull_data_from_sql(naming_pattern, "Team", Team1)
            team2_data = pull_data_from_sql(naming_pattern, "Team", Team2)

            data_difference = subtract_tables(team1_data, team2_data)
            data_difference.to_sql(name=f"z{Team1}_{Team2}_{date_of_game}_table{j}", con=engine, if_exists="replace", index=False)
            j = j + 1







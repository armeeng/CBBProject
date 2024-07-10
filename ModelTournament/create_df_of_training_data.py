import pandas as pd
from sqlalchemy import create_engine

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

dfs = []

# Iterate through each row in the game_results_table_data DataFrame
for index, row in game_results_table_data.iterrows():
    Team1 = row.iloc[0]
    Team2 = row.iloc[2]
    date_of_game = row.iloc[4]
    winner = row.iloc[6]

    # Query SQL to get the difference tables for the matchup
    table1_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table1`"
    table1_data = pd.read_sql(table1_query, engine).select_dtypes(include='number')

    table2_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table2`"
    table2_data = pd.read_sql(table2_query, engine).select_dtypes(include='number')

    table3_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table3`"
    table3_data = pd.read_sql(table3_query, engine).select_dtypes(include='number')

    # Create a DataFrame with the required columns
    df_row = pd.DataFrame({
        'Team1': [Team1],
        'Team2': [Team2],
        'DifferenceTable1': [table1_data],
        'DifferenceTable2': [table2_data],
        'DifferenceTable3': [table3_data],
        'Winner': [winner]
    })

    # Append the DataFrame to the list
    dfs.append(df_row)


# Concatenate all DataFrames in the list into one DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Send the DataFrame to SQL
final_df.to_sql('yMatchup_Differences', engine, if_exists='append', index=False)  # Replace 'your_table_name' with the actual name of your table in the database

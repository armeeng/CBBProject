from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None) # So you can view the whole table
pd.set_option('display.max_columns', None) # So you can view the whole table

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

#schedule_query = "SELECT * FROM Schedule"
#schedule_table_data = pd.read_sql(schedule_query, engine)
#schedule_table_data = schedule_table_data.drop(index=[7, 11, 18])


game_results_query = "SELECT * FROM game_results"
game_results_table_data = pd.read_sql(game_results_query, engine)
print(game_results_table_data)

naming_patterns = ["url1_", "url2_", "url3_"]
summary_data = pd.DataFrame(columns=['Team1', 'Team2', 'DataDifferencesSum', 'MultDataDifferencesSum'])

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

def sum_data_table(data_table):
    # Select numeric values only
    numeric_values = data_table.select_dtypes(include=['number']).to_numpy(na_value=0)

    # Sum all numeric values in the DataFrame
    total_sum = numeric_values.sum()

    return total_sum

def subtract_tables(data_table1, data_table2):
    # Assuming both tables have the same structure
    # Subtract corresponding values from table2 to table1
    numeric_columns = data_table1.select_dtypes(include=['number']).columns
    result_table = data_table1[numeric_columns].subtract(data_table2[numeric_columns])

    # Add the 'Table' column from data_table1 to the result_table
    result_table['Stat'] = data_table1['Stat']

    return result_table

def get_team_better_data(data_difference, num):
    numeric_columns = data_difference.select_dtypes(include=['number']).columns

    # Find the cells that are 0 and store those cells
    zero_cells = data_difference[numeric_columns] == 0

    if num == 1:
        team_better_numeric = data_difference[numeric_columns] >= 0
    elif num == 2:
        team_better_numeric = data_difference[numeric_columns] <= 0
    else:
        raise ValueError("Invalid value for num. Use 1 for Team1 or 2 for Team2.")

    # Replace the boolean values with 0 wherever there were cells with 0's
    team_better_numeric = team_better_numeric.where(~zero_cells, 0)

    team_better = pd.concat([data_difference['Stat'], team_better_numeric], axis=1)

    return team_better

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

def update_records(data_table, num, naming_pattern):
    if num not in {1, 2}:
        raise ValueError("Invalid value for num. Use 1 for Team1 or 2 for Team2.")


    if naming_pattern == "url1_":
        table_name = "url_stat_1"
    elif naming_pattern == "url2_":
        table_name = "url_stat_2"
    elif naming_pattern == "url3_":
        table_name = "url_stat_3"
    else:
        raise ValueError(f"Invalid naming_pattern: {naming_pattern}")

    stat_table = pd.read_sql(f"SELECT * FROM {table_name}", engine)

    # Iterate through each row in data_table
    if num == 1:
        for index, row in data_table.iterrows():
            for col in data_table.columns:
                # Update corresponding cell in stat_table based on the condition in data_table
                if pd.notna(row[col]):
                    if isinstance(row[col], bool):
                        # Use increment_string to update the cell in stat_table
                        stat_table.loc[index, col] = increment_string(stat_table.loc[index, col], 1 if row[col] else 2)
                    elif row[col] == 0:
                        # Do nothing for 0 value in data_table
                        pass
                    else:
                        pass
                else:
                    pass
    elif num == 2:
        for index, row in data_table.iterrows():
            for col in data_table.columns:
                # Update corresponding cell in stat_table based on the condition in data_table
                if pd.notna(row[col]):
                    if isinstance(row[col], bool):
                        # Use increment_string to update the cell in stat_table
                        stat_table.loc[index, col] = increment_string(stat_table.loc[index, col], 2 if row[col] else 1)
                    elif row[col] == 0:
                        # Do nothing for 0 value in data_table
                        pass
                    else:
                        pass
                else:
                    pass

    stat_table.to_sql(name=table_name, con=engine, index=False, if_exists="replace")

def extract_percentage(cell):
    parts = [part.strip() for part in cell.split(',')]
    percentage_part = next((part for part in parts if part.startswith('%:')), None)
    if percentage_part:
        return float(percentage_part.split(': ')[1])
    else:
        return cell

def turn_to_percent_table(data_table):
    percent_table = data_table.map(extract_percentage)
    return percent_table

for index, row in game_results_table_data.iterrows():
    data_differences_sum = 0
    mult_data_differences_sum = 0

    Team1 = row.iloc[0]
    Team2 = row.iloc[2]

    print(Team1)
    print(Team2)

    team1_row_indices = game_results_table_data.loc[(game_results_table_data['Team1'] == Team1)].index
    team2_row_indices = game_results_table_data.loc[(game_results_table_data['Team2'] == Team2)].index
    common_indices = set(team1_row_indices) & set(team2_row_indices)

    # Check if there are common indices before checking 'is_finished'
    if common_indices:
        is_finished = all(game_results_table_data.loc[index, 'Status'] == 'Finished' for index in common_indices)
        # Rest of your code using is_finished
    else:
        # Handle the case when there are no common indices
        is_finished = False

    if is_finished:

        # Determine the winner based on the scores
        score1 = pd.to_numeric(game_results_table_data.loc[next(iter(common_indices)), 'Score1'])
        score2 = pd.to_numeric(game_results_table_data.loc[next(iter(common_indices)), 'Score2'])

        if score1 > score2:
            winner = Team1
        elif score2 > score1:
            winner = Team2



    for naming_pattern in naming_patterns:
        team1_data = pull_data_from_sql(naming_pattern, "Team", Team1)
        team2_data = pull_data_from_sql(naming_pattern, "Team", Team2)

        if team1_data.empty or team2_data.empty:
            input("Empty Dataframe Found...")

        if naming_pattern == "url1_":
            table_name = "url_stat_1"
        elif naming_pattern == "url2_":
            table_name = "url_stat_2"
        elif naming_pattern == "url3_":
            table_name = "url_stat_3"
        else:
            raise ValueError(f"Invalid naming_pattern: {naming_pattern}")

        stat_table = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        percent_table = turn_to_percent_table(stat_table)

        data_difference = subtract_tables(team1_data, team2_data)

        mult_data_difference = percent_table[percent_table.select_dtypes(include=['number']).columns] * data_difference[percent_table.select_dtypes(include=['number']).columns]

        data_difference_sum = sum_data_table(data_difference)
        mult_data_difference_sum = sum_data_table(mult_data_difference)

        data_differences_sum += data_difference_sum
        mult_data_differences_sum += mult_data_difference_sum

        if is_finished:
            team1_better = get_team_better_data(data_difference, 1)

            if winner == Team1:
                updated_stat_table = update_records(team1_better, 1, naming_pattern)
            elif winner == Team2:
                updated_stat_table = update_records(team1_better, 2, naming_pattern)

            game_results_table_data.at[next(iter(common_indices)), 'Status'] = 'Complete'
            game_results_table_data.to_sql(name="game_results", con=engine, index=False, if_exists="replace")

    # Create a dictionary with the values
    new_row = {
        'Team1': Team1,
        'Team2': Team2,
        'DataDifferencesSum': data_differences_sum,
        'MultDataDifferencesSum': mult_data_differences_sum
    }

    # Use loc to add a new row to the DataFrame
    summary_data.loc[len(summary_data)] = new_row

# Print the DataFrame
print(summary_data)






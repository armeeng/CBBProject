from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np
from datetime import datetime
import io



startTime = datetime.now()

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

def create_random_table(lowerBound, upperBound, height, width):
    random_table = pd.DataFrame(np.random.randint(lowerBound, upperBound, size=(height, width)))
    return random_table

def add_noise_to_table(table):
    noise = pd.DataFrame(np.random.normal(0, 0.05, size=table.shape), columns=table.columns)
    return table + noise

def multiply_noise_to_table(table):
    noise = pd.DataFrame(np.random.normal(1, 0.01, size=table.shape), columns=table.columns)
    return table * noise

def divide_noise_to_table(table):
    noise = pd.DataFrame(np.random.normal(0, 100, size=table.shape), columns=table.columns)
    return table / noise

def subtract_noise_to_table(table):
    noise = pd.DataFrame(np.random.normal(0, 2, size=table.shape), columns=table.columns)
    return table - noise

def multiply_table(data_table, random_table):
    # Get the number of columns in each table
    num_columns_data = len(data_table.columns)
    num_columns_random = len(random_table.columns)

    # Initialize a new DataFrame to store the multiplied values
    multiplied_table = pd.DataFrame()

    # Perform element-wise multiplication for each column in the random table
    for i in range(num_columns_random):
        if i < num_columns_data:  # Check if data_table has enough columns
            multiplied_table[i] = data_table.iloc[:, i] * random_table.iloc[:, i]
        else:
            # If data_table has fewer columns, break the loop
            break

    return multiplied_table

def sum_data_table(data_table):
    # Select numeric values only
    numeric_values = data_table.select_dtypes(include=['number']).to_numpy(na_value=0)

    # Sum all numeric values in the DataFrame
    total_sum = numeric_values.sum()

    return total_sum

def store_best_50_accuracies(accuracy, random_tables_map):
    # Create a new inspector
    inspector = inspect(engine)

    # Check if the table exists
    if 'best_50_accuracies' in inspector.get_table_names():
        best_50_df = pd.read_sql('best_50_accuracies', engine)
    else:
        best_50_df = pd.DataFrame(columns=['Accuracy', 'Random Table 1', 'Random Table 2', 'Random Table 3'])

    # Add the new entry
    new_entry = pd.DataFrame([[accuracy, random_tables_map['random_table1'], random_tables_map['random_table2'], random_tables_map['random_table3']]], columns=['Accuracy', 'Random Table 1', 'Random Table 2', 'Random Table 3'])
    best_50_df = best_50_df._append(new_entry, ignore_index=True)

    # Sort by accuracy and keep the top 50
    best_50_df.sort_values(by='Accuracy', ascending=False, inplace=True)
    best_50_df = best_50_df.head(50)

    # Save to SQL
    best_50_df.to_sql('best_50_accuracies', engine, if_exists='replace', index=False)

while(True):
    lower_random_bound = -1
    upper_random_bound = 1

    # Load the best accuracy from the database
    best_50_df = pd.read_sql('best_50_accuracies', engine, index_col='Accuracy')

    # Randomly select a row from the best_50_df DataFrame
    selected_index = np.random.randint(0, 4)
    selected_table = best_50_df.iloc[selected_index]

    # Extract random tables from the selected row
    random_table1 = selected_table['Random Table 1']
    random_table2 = selected_table['Random Table 2']
    random_table3 = selected_table['Random Table 3']

    random_table1 = pd.read_csv(io.StringIO(random_table1), delim_whitespace=True)
    random_table2 = pd.read_csv(io.StringIO(random_table2), delim_whitespace=True)
    random_table3 = pd.read_csv(io.StringIO(random_table3), delim_whitespace=True)

    # Add noise to each table
    random_table1 = add_noise_to_table(random_table1)
    random_table2 = add_noise_to_table(random_table2)
    random_table3 = add_noise_to_table(random_table3)

    random_tables_map = {
        'random_table1': random_table1,
        'random_table2': random_table2,
        'random_table3': random_table3
    }

    correct = 0
    incorrect = 0

    for index, row in game_results_table_data.iterrows():
        data_difference_sum = 0
        difference_sum = 0

        Team1 = row.iloc[0]
        Team2 = row.iloc[2]
        date_of_game = row.iloc[4]
        winner = row.iloc[6]

        table1_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table1`"
        table1_data = pd.read_sql(table1_query, engine)

        table2_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table2`"
        table2_data = pd.read_sql(table2_query, engine)

        table3_query = f"SELECT * FROM `z{Team1}_{Team2}_{date_of_game}_table3`"
        table3_data = pd.read_sql(table3_query, engine)

        random_multiplied_table1 = multiply_table(table1_data, random_table1)
        random_multiplied_table2 = multiply_table(table2_data, random_table2)
        random_multiplied_table3 = multiply_table(table3_data, random_table3)

        data_difference_sum = sum_data_table(random_multiplied_table1) + sum_data_table(random_multiplied_table2) + sum_data_table(random_multiplied_table3)

        if data_difference_sum <= 0:
            predicted_winner = Team2
        else:
            predicted_winner = Team1

        if predicted_winner == winner:
            correct = correct + 1
        else:
            incorrect = incorrect + 1


    accuracy = correct / (correct+incorrect)
    accuracy = abs(accuracy - 0.5)

    print(accuracy)
    store_best_50_accuracies(accuracy, random_tables_map)

    endTime = datetime.now()
    print(endTime - startTime)







import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None) # So you can view the whole table
pd.set_option('display.max_columns', None) # So you can view the whole table

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

# Load data from the database
matchup_df = pd.read_sql('yMatchup_Differences', engine)

def dataframe_to_array(df):
    # Convert DataFrame to a 1D array
    return df.values.flatten()

def convert_to_dataframe(string_table):
    # Read the string table into a DataFrame
    df = pd.read_csv(io.StringIO(string_table), delim_whitespace=True)
    # Keep only numerical columns and drop columns with NaN values
    df = df.select_dtypes(include='number').dropna(axis=1)
    # Set column names to None
    df.columns = [None] * len(df.columns)
    return df

# Apply the function to each string table column
matchup_df['DifferenceTable1'] = matchup_df['DifferenceTable1'].apply(convert_to_dataframe)
matchup_df['DifferenceTable1'] = matchup_df['DifferenceTable1'].apply(lambda x: x.iloc[:, 1:])
matchup_df['DifferenceTable2'] = matchup_df['DifferenceTable2'].apply(convert_to_dataframe)
matchup_df['DifferenceTable2'] = matchup_df['DifferenceTable2'].apply(lambda x: x.iloc[:, 1:])
matchup_df['DifferenceTable3'] = matchup_df['DifferenceTable3'].apply(convert_to_dataframe)

# Combine arrays into a single array
matchup_df['Array'] = matchup_df.apply(lambda row: np.concatenate((dataframe_to_array(row['DifferenceTable1']),
                                                                  dataframe_to_array(row['DifferenceTable2']),
                                                                  dataframe_to_array(row['DifferenceTable3']))), axis=1)



# Encode Winner column as 0 for Team1 wins and 1 for Team2 wins
matchup_df['Winner'] = matchup_df.apply(lambda row: 0 if row['Winner'] == row['Team1'] else 1, axis=1)

# Rearrange columns and drop unnecessary ones
finaldf = matchup_df[['Team1', 'Team2', 'Array', 'Winner']]

while True:
    # Split data into features (X) and target (y)
    X = np.vstack(finaldf['Array'].values)  # Stack the arrays vertically
    y = finaldf['Winner'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the best model
    best_model = tf.keras.models.load_model("FeedforwardNeuralNetwork.keras")

    # Define the FNN model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),  # Example of adding dropout regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Example of adding dropout regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Example of adding dropout regularization
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # Evaluate the model on the testing set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Current Model Test Loss: {loss}, Current Model Test Accuracy: {accuracy}')

    best_loss, best_accuracy = best_model.evaluate(X_test, y_test)
    print(f'Best Model Test Loss: {best_loss}, Best Model Test Accuracy: {best_accuracy}')

    # Check if the current model has better validation accuracy than the best model
    if accuracy > best_accuracy:
        # Save the current model as the best model
        model.save("FeedforwardNeuralNetwork.keras")
        print(f"Best Model Accuracy Updated: {accuracy}")



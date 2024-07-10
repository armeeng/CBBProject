import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

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
                                                                   dataframe_to_array(row['DifferenceTable3']))),
                                       axis=1)

# Encode Winner column as 0 for Team1 wins and 1 for Team2 wins
matchup_df['Winner'] = matchup_df.apply(lambda row: 0 if row['Winner'] == row['Team1'] else 1, axis=1)

# Rearrange columns and drop unnecessary ones
finaldf = matchup_df[['Team1', 'Team2', 'Array', 'Winner']]
finaldf = finaldf[finaldf['Array'].apply(lambda x: x.shape) == (705,)]

# Split data into features (X) and target (y)
X = np.vstack(finaldf['Array'].values)  # Stack the arrays vertically
y = finaldf['Winner'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model-building function for Keras Tuner
def build_model(hp):
    model = tf.keras.models.Sequential()

    # Tune the number of units in the first dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', input_shape=(X_train.shape[1],)))

    # Tune the number of dense layers
    hp_layers = hp.Int('layers', min_value=1, max_value=4, step=1)
    for i in range(hp_layers):
        model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer with a dropout rate of 0.5

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Instantiate the RandomSearch tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',  # Optimize for validation accuracy
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=3,  # Number of models to train per trial
    directory='hyperparameter_tuning',
    project_name='basketball_prediction'
)

while True:
    best_model = tf.keras.models.load_model("FeedforwardNeuralNetworkMore.keras")

    # Perform hyperparameter tuning
    tuner.search(X_train, y_train,
                 epochs=5,             # Number of epochs for each model training
                 validation_split=0.2)  # Fraction of training data to use for validation

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    print(best_hp)

    # Build the best model
    model = tuner.hypermodel.build(best_hp)

    # Train the best model
    model.fit(X_train, y_train, epochs=5, validation_split=0.2)

    # Evaluate the best model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    best_loss, best_accuracy = best_model.evaluate(X_test, y_test)
    print(f'Best Model Test Loss: {best_loss}, Best Model Test Accuracy: {best_accuracy}')

    # Check if the current model has better validation accuracy than the best model
    if accuracy > best_accuracy:
        # Save the current model as the best model
        model.save("FeedforwardNeuralNetworkMore.keras")
        print(f"Best Model Accuracy Updated: {accuracy}")

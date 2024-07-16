## Overview

This project involves scraping data from multiple pages on the Team Rankings website and storing it in a MySQL database. The data includes information such as team rankings, statistics, and schedules. The scripts are organized into different directories based on the type of data being scraped (e.g., `Current`, `RandomTournament`, `Historical`).

## Models

### Current Model

- A non-neural network model.
- Based on games played in the current season.

### Historical Model

- A non-neural network model.
- Based on all games since 2011.

### ModelSeason

- A neural network model.
- Trained on regular season games since 2011.

### ModelTournament

- A neural network model.
- Trained on post-season games (conference tournaments, CBI, NIT, March Madness).

### RandomSeason

- A non-neural network model.
- Uses regular season games and a random matrix multiplier.
- Random matrices are generated and tested using historical data.
- Matrices meeting a certain accuracy threshold are stored in the database along with their corresponding accuracy on the historical data.

### RandomTournament

- A non-neural network model.
- Uses only post-season games (conference tournaments, CBI, NIT, March Madness).
- The process is similar to RandomSeason.
- Random matrices are generated and tested using historical tournament data.
- Matrices meeting a certain accuracy threshold are stored in the database along with their corresponding accuracy on the historical tournament data.

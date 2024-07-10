from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO
import os
from sklearn.preprocessing import MinMaxScaler
import re

pd.set_option('display.max_rows', None) # So you can view the whole table
pd.set_option('display.max_columns', None) # So you can view the whole table

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

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

stat_list1 = [os.path.basename(url) for url in url_list1]
stat_list2 = [os.path.basename(url) for url in url_list2]
stat_list3 = [os.path.basename(url) for url in url_list3]


# Create DataFrames
df1 = pd.DataFrame({'Stat': stat_list1})
df2 = pd.DataFrame({'Stat': stat_list2})
df3 = pd.DataFrame({'Stat': stat_list3})


df1['2022'] = "W: 0, L: 0, %: 0"
df1['Last 3'] = "W: 0, L: 0, %: 0"
df1['Last 1'] = "W: 0, L: 0, %: 0"
df1['Home'] = "W: 0, L: 0, %: 0"
df1['Away'] = "W: 0, L: 0, %: 0"
df1['2021'] = "W: 0, L: 0, %: 0"

df2['Rating'] = "W: 0, L: 0, %: 0"
df2['v 1-25'] = "W: 0, L: 0, %: 0"
df2['v 26-50'] = "W: 0, L: 0, %: 0"
df2['v 51-100'] = "W: 0, L: 0, %: 0"

df3['Rating'] = "W: 0, L: 0, %: 0"

df1.sort_values(by='Stat', ascending=True, inplace=True)
df2.sort_values(by='Stat', ascending=True, inplace=True)
df3.sort_values(by='Stat', ascending=True, inplace=True)




#df1.to_sql(name="url_stat_1", con=engine, if_exists='fail', index=False)
#df2.to_sql(name="url_stat_2", con=engine, if_exists='fail', index=False)
#df3.to_sql(name="url_stat_3", con=engine, if_exists='fail', index=False)



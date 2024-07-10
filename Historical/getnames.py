from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
from io import StringIO
import os
from sklearn.preprocessing import MinMaxScaler
import re

startTime = datetime.now()

ESPNNames = ["Abilene Christian",
"Air Force",
"Akron",
"Alabama A&M",
"Alabama",
"Alabama State",
"Albany",
"Alcorn State",
"American University",
"Appalachian State",
"Arizona State",
"Arizona",
"Arkansas",
"Arkansas State",
"Arkansas-Pine Bluff",
"Army",
"Auburn",
"Austin Peay",
"BYU",
"Ball State",
"Baylor",
"Bellarmine",
"Belmont",
"Bethune-Cookman",
"Binghamton",
"Boise State",
"Boston College",
"Boston University",
"Bowling Green",
"Bradley",
"Brown",
"Bryant",
"Bucknell",
"Buffalo",
"Butler",
"Cal Poly",
"Cal State Bakersfield",
"Cal State Fullerton",
"Cal State Northridge",
"California Baptist",
"California",
"Campbell",
"Canisius",
"Central Arkansas",
"Central Connecticut",
"Central Michigan",
"Charleston",
"Charleston Southern",
"Charlotte",
"Chattanooga",
"Chicago State",
"Cincinnati",
"Clemson",
"Cleveland State",
"Coastal Carolina",
"Colgate",
"Colorado",
"Colorado State",
"Columbia",
"Coppin State",
"Cornell",
"Creighton",
"Dartmouth",
"Davidson",
"Dayton",
"DePaul",
"Delaware",
"Delaware State",
"Denver",
"Detroit",
"Drake",
"Drexel",
"Duke",
"Duquesne",
"East Carolina",
"East Tennessee State",
"Eastern Illinois",
"Eastern Kentucky",
"Eastern Michigan",
"Eastern Washington",
"Elon",
"Evansville",
"Fairfield",
"Fairleigh Dickinson",
"Florida A&M",
"Florida Atlantic",
"Florida",
"Florida Gulf Coast",
"Florida International",
"Florida State",
"Fordham",
"Fresno State",
"Furman",
"Gardner-Webb",
"George Mason",
"George Washington",
"Georgetown",
"Georgia",
"Georgia Southern",
"Georgia State",
"Georgia Tech",
"Gonzaga",
"Grambling",
"Grand Canyon",
"Green Bay",
"Hampton",
"Harvard",
"Hawai'i",
"High Point",
"Hofstra",
"Holy Cross",
"Houston Christian",
"Houston",
"Howard",
"IUPUI",
"Idaho State",
"Idaho",
"Illinois",
"Illinois State",
"Incarnate Word",
"Indiana",
"Indiana State",
"Iona",
"Iowa",
"Iowa State",
"Jackson State",
"Jacksonville",
"Jacksonville State",
"James Madison",
"Kansas City",
"Kansas",
"Kansas State",
"Kennesaw State",
"Kent State",
"Kentucky",
"LSU",
"La Salle",
"Lafayette",
"Lamar",
"Lehigh",
"Liberty",
"Lindenwood",
"Lipscomb",
"Little Rock",
"Long Beach State",
"Long Island University",
"Longwood",
"Louisiana",
"Louisiana Tech",
"Louisville",
"Loyola Chicago",
"Loyola Maryland",
"Loyola Marymount",
"Maine",
"Manhattan",
"Marist",
"Marquette",
"Marshall",
"Maryland",
"Maryland-Eastern Shore",
"McNeese",
"Memphis",
"Mercer",
"Merrimack",
"Miami (OH)",
"Miami",
"Michigan State",
"Michigan",
"Middle Tennessee",
"Milwaukee",
"Minnesota",
"Mississippi State",
"Mississippi Valley State",
"Missouri State",
"Missouri",
"Monmouth",
"Montana",
"Montana State",
"Morehead State",
"Morgan State",
"Mount St. Mary's",
"Murray State",
"NC State",
"NJIT",
"Navy",
"Nebraska",
"Nevada",
"New Hampshire",
"New Mexico",
"New Mexico State",
"New Orleans",
"Niagara",
"Nicholls",
"Norfolk State",
"North Alabama",
"North Carolina A&T",
"North Carolina Central",
"North Carolina",
"North Dakota",
"North Dakota State",
"North Florida",
"North Texas",
"Northeastern",
"Northern Arizona",
"Northern Colorado",
"Northern Illinois",
"Northern Iowa",
"Northern Kentucky",
"Northwestern State",
"Northwestern",
"Notre Dame",
"Oakland",
"Ohio",
"Ohio State",
"Oklahoma",
"Oklahoma State",
"Old Dominion",
"Ole Miss",
"Omaha",
"Oral Roberts",
"Oregon",
"Oregon State",
"Pacific",
"Penn State",
"Pennsylvania",
"Pepperdine",
"Pittsburgh",
"Portland",
"Portland State",
"Prairie View A&M",
"Presbyterian",
"Princeton",
"Providence",
"Purdue",
"Purdue Fort Wayne",
"Queens University",
"Quinnipiac",
"Radford",
"Rhode Island",
"Rice",
"Richmond",
"Rider",
"Robert Morris",
"Rutgers",
"SE Louisiana",
"SIU Edwardsville",
"SMU",
"Sacramento State",
"Sacred Heart",
"Saint Joseph's",
"Saint Louis",
"Saint Mary's",
"Saint Peter's",
"Sam Houston",
"Samford",
"San Diego State",
"San Diego",
"San Francisco",
"San José State",
"Santa Clara",
"Seattle U",
"Seton Hall",
"Siena",
"South Alabama",
"South Carolina",
"South Carolina State",
"South Carolina Upstate",
"South Dakota",
"South Dakota State",
"South Florida",
"Southeast Missouri State",
"Southern Illinois",
"Southern Indiana",
"Southern",
"Southern Miss",
"Southern Utah",
"St. Bonaventure",
"St. Francis (PA)",
"St. Francis Brooklyn",
"St. John's",
"St. Thomas - Minnesota",
"Stanford",
"Stephen F. Austin",
"Stetson",
"Stonehill",
"Stony Brook",
"Syracuse",
"TCU",
"Tarleton",
"Temple",
"Tennessee State",
"Tennessee Tech",
"Tennessee",
"Texas A&M",
"Texas A&M-Commerce",
"Texas A&M-Corpus Christi",
"Texas",
"Texas Southern",
"Texas State",
"Texas Tech",
"The Citadel",
"Toledo",
"Towson",
"Troy",
"Tulane",
"Tulsa",
"UAB",
"UC Davis",
"UC Irvine",
"UC Riverside",
"UC San Diego",
"UC Santa Barbara",
"UCF",
"UCLA",
"UConn",
"UIC",
"UL Monroe",
"UMBC",
"UMass Lowell",
"UMass",
"UNC Asheville",
"UNC Greensboro",
"UNC Wilmington",
"UNLV",
"USC",
"UT Arlington",
"UT Martin",
"UT Rio Grande Valley",
"UTEP",
"UTSA",
"Utah State",
"Utah Tech",
"Utah",
"Utah Valley",
"VCU",
"VMI",
"Valparaiso",
"Vanderbilt",
"Vermont",
"Villanova",
"Virginia",
"Virginia Tech",
"Wagner",
"Wake Forest",
"Washington",
"Washington State",
"Weber State",
"West Virginia",
"Western Carolina",
"Western Illinois",
"Western Kentucky",
"Western Michigan",
"Wichita State",
"William & Mary",
"Winthrop",
"Wisconsin",
"Wofford",
"Wright State",
"Wyoming",
"Xavier",
"Yale",
"Youngstown State"]

# MySQL Database Connection Details
####removed

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)
url = "https://www.teamrankings.com/ncaa-basketball/stat/points-per-game?date=2023-12-27"

WebContentsNorm = requests.get(url)

ReadContentsNorm = BeautifulSoup(WebContentsNorm.content, "lxml")

FindTableNorm = ReadContentsNorm.find("table")

PresentTableNorm = pd.read_html(StringIO(str(FindTableNorm)), flavor="lxml")[0]
PresentTableNorm = PresentTableNorm.iloc[:, 1]
PresentTableNorm['lowercase_sort'] = PresentTableNorm.str.lower()
PresentTableNorm = PresentTableNorm.sort_values(key=lambda x: x.str.lower())
PresentTableNorm = PresentTableNorm.drop(index=PresentTableNorm.index[-1])

# Creating a new DataFrame with ESPNNames
espn_df = pd.DataFrame({'ESPNNames': ESPNNames})
PresentTableNorm.reset_index(drop=True, inplace=True)
espn_df.reset_index(drop=True, inplace=True)

result_df = pd.concat([PresentTableNorm, espn_df], axis=1)

# Print the result DataFrame
print(result_df)


result_df.to_sql(name="team_names", con=engine, if_exists="replace", index=False)

endTime = datetime.now()
print(endTime - startTime)


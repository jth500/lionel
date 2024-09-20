# lionel
A Fantasy Premier League team picking tool.

## Index

- [About](#about)
- [Usage](#usage)
  - [Installation](#installation)
  - [Commands](#commands)
- [Data](#Data)
- [License](#license)

## About
Lionel predicts Fantasy Premier League (FPL) points using a Bayesian hierarchical model, implemented with a No-U-Turn Sampler in PyMC. Then, it maximises the predicted points using linear programming as implemented in PuLP.

## Usage

### Installation

```
$ TODO: Installation steps
```

### Example usage

Using FPLPointsModel to get expected points
```
import pandas as pd
import numpy as np
import arviz as az

from lionel.model.hierarchical import FPLPointsModel

player = ["player_1", "player_2", "player_3", "player_4", "player_5", "player_6"] 
gameweek = [1] * len(player) + [2] * len(player)
season = [25] * len(player) * 2
home_team = ["team_1"] * len(player) + ["team_2"] * len(player)
away_team = ["team_2"] * len(player) + ["team_1"] * len(player)
home_goals = [1] * len(player) + [2] * len(player)
away_goals = [0] * len(player) + [1] * len(player)
position = ["FWD", "MID", "DEF", "GK", "FWD", "MID"] * 2
minutes = [90] * len(player) * 2
goals_scored = [1, 0, 0, 0, 0, 0] + [0, 0, 1, 1, 0, 1]
assists = [0, 1, 0, 0, 0, 0] + [1, 0, 0, 0, 1, 1]
is_home = [True, True, True, False, False, False] + [False, False, False, True, True, True]
points = [10, 6, 2, 2, 2, 2] + [6, 2, 10, 10, 2, 10]

df = pd.DataFrame({
    'player': player + player, 
    'gameweek': gameweek, 
    'season': season, 
    'home_team': home_team, 
    'away_team': away_team, 
    'home_goals': home_goals, 
    'away_goals': away_goals, 
    'position': position, 
    'minutes': minutes, 
    'goals_scored': goals_scored, 
    'assists': assists, 
    'is_home': is_home
})

fplm = FPLPointsModel()
fplm.fit(df, points)

# Get predicted points
fplm.predict(df, extend_idata=False, combined=False, predictions=True)

# array([8.1046909 , 4.69126912, 6.41898185, 8.3587504 , 2.86888174,6.79323443, 7.11748533, 3.33156234, 4.50141007, 6.13532871, 1.75916264, 5.61056724])
```

Using selector to select a Fantasy Team 

```
import pandas as pd
from lionel.selector import NewXVSelector, XISelector

df = pd.DataFrame({
    'player': [f"player_{i}" for i in range(20)],
    'team_name': [f'team_{i}' for i in range(10)] * 2,
    'position': ["FWD"] * 3 + ["MID"] * 8 + ["DEF"] * 7 + ["GK"] * 2,
    'value': [100, 90, 55, 45] * 5,
    'points_pred': np.random.normal(5, 2, 20),
    'xv': [0] * 20,
    'xi': [0] * 20,
    'captain': [0] * 20,
})

xv_sel = NewXVSelector('points_pred')
xv_sel.build_problem(df)
xv_sel.solve()


xi_sel = XISelector("points_pred")
xi_sel.build_problem(xv_sel.data)
xi_sel.solve()

xi_sel.data.sort_values(by='points_pred', ascending=False).head()
```
Returns:
| player      | team_name | position | value | points_pred | xv | xi | captain |
|-------------|-----------|----------|-------|-------------|----|----|---------|
| player_14   | team_4    | DEF      | 55    | 10.779401   | 1  | 1  | 1       |
| player_2    | team_2    | FWD      | 55    | 7.780676    | 1  | 1  | 0       |
| player_19   | team_9    | GK       | 45    | 7.769195    | 1  | 1  | 0       |
| player_4    | team_4    | MID      | 100   | 7.053329    | 0  | 0  | 0       |
| player_13   | team_3    | DEF      | 90    | 6.899006    | 1  | 1  | 0       |


## Data
FPL (Pre-2024/25): [Vaastav](https://github.com/vaastav/Fantasy-Premier-League)  
Betting: [The Odds API](https://the-odds-api.com)


##  License
MIT





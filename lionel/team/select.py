import os
import sys
from pathlib import Path
import numpy as np
import datetime as dt
import pandas as pd
from pulp import LpVariable, LpProblem, lpSum, LpMaximize
from abc import abstractmethod, ABC

# Add root to path when module is run as a script
ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.append(os.path.dirname(str(ROOT)))

from lionel.utils import setup_logger


# TODO: Refactor this so that the object is instantiated once
# and the methods can be called on each prediction variable
logger = setup_logger(__name__)


class BaseSelector(ABC):

    def __init__(self, player_df, season, pred_var="pred_Naive"):
        self.player_df = player_df.fillna(0).reset_index(drop=True)
        self.season = season
        self.first_xi = pd.DataFrame()
        self.players = None
        self.pred_var = pred_var

    @property
    @abstractmethod
    def POS_CONSTRAINTS():
        pass

    @abstractmethod
    def _add_position_constraints():
        pass

    @staticmethod
    def _create_decision_var(prefix, df):
        return [LpVariable(prefix + str(i), cat="Binary") for i in df.index]

    @staticmethod
    def _get_player_indices(players) -> list:
        players = [player for player in players if player.varValue != 0]
        indices = [int(player.name.split("_")[1]) for player in players]
        return indices


class XISelector(BaseSelector):
    POS_CONSTRAINTS = {
        "DEF": [5, 3],
        "FWD": [3, 1],
        "MID": [5, 2],
        "GK": [1, 1],
    }

    def __init__(self, player_df, season, pred_var):
        super().__init__(player_df, season, pred_var)
        self.first_xv = pd.DataFrame()
        self.other_players = pd.DataFrame()
        self.players = self._create_decision_var("player_", self.first_xv)
        logger.debug("Initialising XI selector object")

    @property
    def first_xv(self):
        if self._first_xv.empty:
            self._first_xv = self.player_df[self.player_df.picked == 1]
        return self._first_xv

    @first_xv.setter
    def first_xv(self, val):
        self._first_xv = val

    @property
    def other_players(self):
        if self._other_players.empty:
            self._other_players = self.player_df[self.player_df.picked == 0]
        return self._other_players

    @other_players.setter
    def other_players(self, val):
        self._other_players = val

    def _initialise_xi_prob(self):
        prob = LpProblem("First team choices", LpMaximize)
        points_weighted = self.first_xv[self.pred_var].to_list()
        prob += lpSum(
            self.players[i] * points_weighted[i] for i in range(len(self.first_xv))
        )
        prob += sum(self.players) == 11
        prob = self._add_position_constraints(prob)
        return prob

    def _add_position_constraints(self, prob):
        positions = self.first_xv.position.to_list()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            # Add upper bound for position
            prob += (
                lpSum(
                    self.players[i]
                    for i in range(len(self.first_xv))
                    if positions[i] == pos
                )
                <= self.POS_CONSTRAINTS[pos][0]
            )
            # Add lower bound for the position
            prob += (
                lpSum(
                    self.players[i]
                    for i in range(len(self.first_xv))
                    if positions[i] == pos
                )
                >= self.POS_CONSTRAINTS[pos][1]
            )
        return prob

    # TODO: Clean up this code
    def _clean_xi(self, indices):
        team = self.first_xv
        team["first_xi"] = 0
        team.loc[indices, "first_xi"] = 1
        team = team.sort_values("first_xi", ascending=False)
        # team = team.drop("index", axis=1).reset_index(drop=True)
        team = pd.concat([team, self.other_players])
        team["season"] = self.season
        team["picked_time"] = dt.datetime.now()
        for col in ["is_home1", "is_home2", "is_home3"]:
            if col in team.columns:
                team[col] = team[col].replace({0: np.nan, float(0): np.nan})
        return team

    def pick_xi(self):
        prob = self._initialise_xi_prob()
        prob.solve()
        indices = self._get_player_indices(self.players)
        team = self._clean_xi(indices)
        self.first_xi = team
        return team


class NewXVSelector(BaseSelector):
    """
    Base class for selecting a team of 15 players. Inherited by UpdateXVSelector,
    which is used to update an existing team.
    """

    # Want to be able to access these objects without instantiating the class
    POS_CONSTRAINTS = {
        "DEF": 5,
        "FWD": 3,
        "MID": 5,
        "GK": 2,
    }
    XI_SELECTOR_OBJ = XISelector

    def __init__(
        self,
        player_df,
        season,
        pred_var,
        budget=1000,
    ):
        super().__init__(player_df, season, pred_var)
        self.budget = budget

        self.first_xv = pd.DataFrame()
        self.teams = self.player_df.team_name.to_list()
        self.positions = self.player_df.position.to_list()
        self.points_weighted = self.player_df[
            self.pred_var
        ].to_list()  # will this be an issue?

        self.players = self._create_decision_var("player_", self.player_df)
        self.captains = self._create_decision_var("captain_", self.player_df)

        self.xi_selector = None
        logger.debug("Initialising XV selector object")

    @property
    def first_xv(self):
        if self._first_xv.empty:
            self._first_xv = self.pick_xv()
        return self._first_xv

    @first_xv.setter
    def first_xv(self, val):
        self._first_xv = val

    @property
    def xi_selector(self):
        if self._xi_selector is None:
            self._xi_selector = NewXVSelector.XI_SELECTOR_OBJ(
                self.first_xv, self.season, self.pred_var
            )
        return self._xi_selector

    @xi_selector.setter
    def xi_selector(self, val):
        self._xi_selector = val

    def _add_budget_constraints(self, prob):
        prob += (
            lpSum(
                self.players[i] * self.player_df.value[self.player_df.index[i]]
                for i in range(len(self.player_df))
            )
            <= self.budget
        )
        return prob

    def _add_club_constraints(self, prob):
        for club in self.teams:
            prob += (
                lpSum(
                    self.players[i]
                    for i in range(len(self.player_df))
                    if self.teams[i] == club
                )
                <= 3
            )
        return prob

    def _add_captain_constraints(self, prob):
        for i in range(len(self.player_df)):
            prob += (self.players[i] - self.captains[i]) >= 0
        return prob

    @staticmethod
    def _get_captain_index(captains) -> int:
        captain = [player for player in captains if player.varValue != 0]
        captain_index = int(captain[0].name.split("_")[1])
        return captain_index

    def _add_position_constraints(self, prob, players, df):
        positions = df.position.to_list()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            prob += (
                lpSum(players[i] for i in range(len(df)) if positions[i] == pos)
                <= self.POS_CONSTRAINTS[pos]
            )
        return prob

    def _clean_xv(self, indices, captain_index):
        team_2 = self.player_df.copy(deep=True)
        team_2.loc[indices, "picked"] = 1
        team_2.loc[team_2["picked"] != 1, "picked"] = 0
        team_2["captain"] = 0
        team_2.loc[captain_index, "captain"] = 1
        return team_2

    def pick_xi(self):
        self.pick_xv()
        self.first_xi = self.xi_selector.pick_xi()
        return self.first_xi

    def _finalise_xv(self):
        # Get indices of selected players and captain
        indices = self._get_player_indices(self.players)
        captain_index = self._get_captain_index(self.captains)
        team = self._clean_xv(indices, captain_index)
        self.first_xv = team
        return self.first_xv

    def _add_xv_constraints(self, prob):
        prob += sum(self.players) == 15
        prob = self._add_budget_constraints(prob)
        prob += sum(self.captains) == 1
        prob = self._add_position_constraints(prob, self.players, self.player_df)
        prob = self._add_club_constraints(prob)
        prob = self._add_captain_constraints(prob)
        return prob

    def initialise_xv_prob(self, *args, **kwargs):
        prob = LpProblem("FPL Player Choices", LpMaximize)
        prob += lpSum(
            (self.players[i] + self.captains[i])
            * self.player_df[self.pred_var][i]  # made a change here
            for i in range(len(self.player_df))
        )
        prob = self._add_xv_constraints(prob)
        return prob

    def pick_xv(self, *args, **kwargs):
        prob = self.initialise_xv_prob(*args, **kwargs)
        prob.solve()
        team = self._finalise_xv()
        return team


class UpdateXVSelector(NewXVSelector):
    # TODO: Add budget change logic
    def __init__(self, player_df, season, initial_xi, budget=1500):
        self.inital_xi_added = False
        self.initial_xi = initial_xi
        super().__init__(player_df, season, budget)
        self.budgeter = None
        logger.debug("Initialising update selector object")

    # @property
    # def budgeter(self):
    #     if self._budgeter is None:
    #         try:
    #             picks = get_my_team_info()["picks"]
    #             self._budgeter = Budgeter(picks)
    #         except ValueError:
    #             logger.info("Cannot create budgeter object. No login info.")
    #             pass
    #     return self._budgeter

    # @budgeter.setter
    # def budgeter(self, val):
    #     self._budgeter = val

    @property
    def player_df(self):
        # add initial team to player_df if not already
        if not self.inital_xi_added:
            self._player_df["initial_xi"] = self._player_df["element"].isin(
                self.initial_xi
            )
            self.inital_xi_added = True
        return self._player_df

    @player_df.setter
    def player_df(self, val):
        self._player_df = val

    def _add_changes_constraint(self, prob, max_changes):
        prob += (
            lpSum(
                self.players[i]
                for i in range(len(self.player_df))
                if not self.player_df["initial_xi"][i]
            )
            <= max_changes
        )
        return prob

    def initialise_xv_prob(self, max_changes=1):
        # add to method from parent class
        prob = super().initialise_xv_prob()
        prob = self._add_changes_constraint(prob, max_changes)
        return prob

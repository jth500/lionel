"""
Module for team selection algorithms.

This module contains the base class and implementations for various team selection algorithms.
It uses linear programming to optimize team selection based on different prediction variables.

Classes:
    BaseSelector: Abstract base class for team selection algorithms.
    NewXVSelector: Concrete implementation of BaseSelector for selecting the first XI players.

Functions:
    setup_logger: Sets up the logger for the module.

Dependencies:
    - pulp
    - abc
    - lionel.utils

Usage:
    This module is intended to be used as part of the larger lionel package for team selection.
    Instantiate a concrete selector class and use its methods to perform team selection.
"""

from pulp import LpVariable, LpProblem, lpSum, LpMaximize
from abc import abstractmethod, ABC
from lionel.utils import setup_logger

logger = setup_logger(__name__)


class BaseSelector(ABC):
    """
    Base class for team selection algorithms.

    Attributes:
        pred_var (str): The name of the variable used for prediction.
        expected_vars (list): A list of expected variables in the input data.
        prob (LpProblem): The optimization problem to be solved.

    Methods:
        validate_inputs(data): Validates the input data.
        build_problem(): Builds the optimization problem.
        solve(): Solves the optimization problem.
        POS_CONSTRAINTS(): Abstract property for position constraints.
        _add_position_constraints(): Abstract method for adding position constraints.
    """

    def __init__(self, pred_var="mean_points_pred"):
        """
        Initializes a BaseSelector object.

        Args:
            pred_var (str, optional): The name of the variable used for prediction. Defaults to "pred_Naive".
        """
        self.pred_var = pred_var
        self.expected_vars = ["player", "team_name", "position", "value", self.pred_var]
        self.prob = None

    def validate_inputs(self, data):
        """
        Validates the input data.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            bool: True if the input data is valid, False otherwise.
        """
        assert all([col in data.columns.to_list() for col in self.expected_vars])
        assert data.player.nunique() == len(data)
        return True

    @abstractmethod
    def build_problem(self):
        """
        Builds the optimization problem.

        Returns:
            LpProblem: The optimization problem.
        """
        pass

    @abstractmethod
    def solve(self):
        """
        Solves the optimization problem.

        Returns:
            pd.DataFrame: The solved optimization problem.
        """
        return None

    @property
    @abstractmethod
    def POS_CONSTRAINTS():
        """
        Abstract property for position constraints.
        """
        pass

    @abstractmethod
    def _add_position_constraints():
        """
        Abstract method for adding position constraints.
        """
        pass


class XISelector(BaseSelector):
    """
    Class for selecting a team of 11 players.

    Attributes:
        POS_CONSTRAINTS (dict): Position constraints for team selection.

    Methods:
        __init__(pred_var): Initializes an XISelector object.
        build_problem(data): Builds the optimization problem for team selection.
        solve(): Solves the optimization problem and returns the selected team.
        _add_position_constraints(prob): Adds position constraints to the optimization problem.
    """

    POS_CONSTRAINTS = {
        "DEF": [5, 3],
        "FWD": [3, 1],
        "MID": [5, 2],
        "GK": [1, 1],
    }

    def __init__(self, pred_var):
        """
        Initializes an XISelector object.

        Args:
            pred_var (str): The name of the variable used for prediction.
        """
        super().__init__(pred_var)
        logger.debug("Initialising XI selector object")

    def build_problem(self, data):
        """
        Builds the optimization problem for team selection.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            LpProblem: The optimization problem.
        """
        assert self.validate_inputs(data)
        assert all(col in data.columns for col in ["picked", "captain"])
        data["first_xi"] = 0

        self.data = data
        self.xv = data[data.picked == 1]
        self.y = self.xv[self.pred_var].values
        assert self.xv.shape[0] == 15

        self.players = [LpVariable(str(i), cat="Binary") for i in self.xv.index]
        prob = LpProblem("First team choices", LpMaximize)

        prob += lpSum(self.players[i] * self.y[i] for i in range(len(self.xv)))
        prob += sum(self.players) == 11
        prob = self._add_position_constraints(prob)
        self.prob = prob
        return self.prob

    def solve(self):
        """
        Solves the optimization problem and returns the selected team.

        Returns:
            pd.DataFrame: The selected team.
        """
        self.prob.solve()
        vals = [(p.varValue, p.name) for p in self.prob.variables()]
        picked_idx = [int(v[1]) for v in vals if v[0] == 1]
        self.xv.loc[self.xv.index.isin(picked_idx), "first_xi"] = 1
        self.data.loc[self.data.index.isin(picked_idx), "first_xi"] = 1
        return self.data

    def _add_position_constraints(self, prob):
        """
        Adds position constraints to the optimization problem.

        Args:
            prob (LpProblem): The optimization problem.

        Returns:
            LpProblem: The optimization problem with added position constraints.
        """
        positions = self.xv.position.to_list()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            prob += (
                lpSum(
                    self.players[i] for i in range(len(self.xv)) if positions[i] == pos
                )
                <= self.POS_CONSTRAINTS[pos][0]
            )
            prob += (
                lpSum(
                    self.players[i] for i in range(len(self.xv)) if positions[i] == pos
                )
                >= self.POS_CONSTRAINTS[pos][1]
            )
        return prob


class NewXVSelector(BaseSelector):
    """
    Base class for selecting a team of 15 players.

    Attributes:
        POS_CONSTRAINTS (dict): Position constraints for team selection.

    Methods:
        __init__(pred_var, budget): Initializes a NewXVSelector object.
        build_problem(data): Builds the optimization problem for team selection.
        solve(): Solves the optimization problem and returns the selected team.
        _add_budget_constraints(prob): Adds budget constraints to the optimization problem.
        _add_club_constraints(prob): Adds club constraints to the optimization problem.
        _add_captain_constraints(prob): Adds captain constraints to the optimization problem.
        _add_position_constraints(prob, players, df): Adds position constraints to the optimization problem.
        _add_xv_constraints(prob): Adds XV constraints to the optimization problem.
    """

    POS_CONSTRAINTS = {
        "DEF": 5,
        "FWD": 3,
        "MID": 5,
        "GK": 2,
    }

    def __init__(
        self,
        pred_var,
        budget=1000,
    ):
        """
        Initializes a NewXVSelector object.

        Args:
            pred_var (str): The name of the variable used for prediction.
            budget (int, optional): The budget for team selection. Defaults to 1000.
        """
        super().__init__(pred_var)
        self.budget = budget
        logger.debug("Initialising XV selector object")

    def build_problem(self, data):
        """
        Builds the optimization problem for team selection.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            LpProblem: The optimization problem.
        """
        assert self.validate_inputs(data)

        data["picked"] = 0

        self.data = data
        self.y = data[self.pred_var].values

        self.players = [LpVariable("p_" + str(i), cat="Binary") for i in data.index]
        self.captains = [LpVariable("c_" + str(i), cat="Binary") for i in data.index]
        self.teams = self.data.team_name.values
        self.values = self.data.value.values

        prob = LpProblem("FPL Player Choices", LpMaximize)
        j = self.data.columns.get_loc(self.pred_var)
        prob += lpSum(
            (self.players[i] + self.captains[i]) * self.data.iloc[i, j]
            for i in range(len(self.players))
        )

        # Add constraints
        prob += sum(self.players) == 15
        prob = self._add_budget_constraints(prob)
        prob += sum(self.captains) == 1
        prob = self._add_position_constraints(prob, self.players, self.data)
        prob = self._add_club_constraints(prob)
        prob = self._add_captain_constraints(prob)
        self.prob = prob
        return self.prob

    def solve(self):
        """
        Solves the optimization problem and returns the selected team.

        Returns:
            pd.DataFrame: The selected team.
        """
        self.prob.solve()
        xv_idx = [int(v.name.split("_")[1]) for v in self.players if v.varValue == 1]
        cap_idx = [int(v.name.split("_")[1]) for v in self.captains if v.varValue == 1]
        self.data.loc[self.data.index.isin(xv_idx), "picked"] = 1
        self.data.loc[self.data.index.isin(cap_idx), "captain"] = 1
        return self.data

    def _add_budget_constraints(self, prob):
        """
        Adds budget constraints to the optimization problem.

        Args:
            prob (LpProblem): The optimization problem.

        Returns:
            LpProblem: The optimization problem with added budget constraints.
        """
        prob += (
            lpSum(self.players[i] * self.values[i] for i in range(len(self.data)))
            <= self.budget
        )
        return prob

    def _add_club_constraints(self, prob):
        """
        Adds club constraints to the optimization problem.

        Args:
            prob (LpProblem): The optimization problem.

        Returns:
            LpProblem: The optimization problem with added club constraints.
        """
        for club in self.teams:
            prob += (
                lpSum(
                    self.players[i]
                    for i in range(len(self.data))
                    if self.teams[i] == club
                )
                <= 3
            )
        return prob

    def _add_captain_constraints(self, prob):
        """
        Adds captain constraints to the optimization problem.

        Args:
            prob (LpProblem): The optimization problem.

        Returns:
            LpProblem: The optimization problem with added captain constraints.
        """
        for i in range(len(self.data)):
            prob += (self.players[i] - self.captains[i]) >= 0
        return prob

    def _add_position_constraints(self, prob, players, df):
        """
        Adds position constraints to the optimization problem.

        Args:
            prob (LpProblem): The optimization problem.
            players (list): The list of player variables.
            df (pd.DataFrame): The input data.

        Returns:
            LpProblem: The optimization problem with added position constraints.
        """
        positions = df.position.to_list()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            prob += (
                lpSum(players[i] for i in range(len(df)) if positions[i] == pos)
                <= self.POS_CONSTRAINTS[pos]
            )
        return prob

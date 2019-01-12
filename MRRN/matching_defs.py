from enum import Enum


class Weighting(Enum):
    NONE = "N"
    EUCLIDEAN = "E"

class Space(Enum):
    COVARIATE = "X"
    PROPENSITY = "P"


class Group():
    def __init__(self, t, lower, upper):
        self.lower = lower
        self.upper = upper
        self.t = t


class MatchingSetup():
    def __init__(self, weighting, matching_space, k_max, levels=1, rem_outliers=True):
        """
        Specifies a matching procedure.
        :param weighting: a Weighting enum
        :param matching_space: space to match in
        :param k_max: max number of neighbours to consider
        :param levels: number of subranges to separate the dose range into
        :param rem_outliers: remove outliers
        """
        self.weighting = weighting
        self.matching_space = matching_space
        self.k_max = k_max
        self.levels = levels
        self.rem_outliers = rem_outliers

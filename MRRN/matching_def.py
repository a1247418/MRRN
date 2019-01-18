from enum import Enum

import loss_def


class Weighting(Enum):
    NONE = "N"
    EUCLIDEAN = "E"

class Space(Enum):
    COVARIATE = "X"
    PROPENSITY = "P"
    PROPENSITY_MEAN = "PM"
    KL = "KL"
    NONE = "N"


class Group():
    def __init__(self, t, lower, upper):
        self.lower = lower
        self.upper = upper
        self.t = t


class MatchingSetup():
    def __init__(self, matching_space, weighting, levels, k_max, needs_propensity_model=False, rem_outliers=True):
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
        self.rem_outliers = int(rem_outliers)

        self.needs_propensity_model = loss_def.needs_propensity(matching_space.value+"_")

    def to_string(self):
        string = "_".join([self.matching_space.value, self.weighting.value, str(self.levels), str(self.k_max),
                           str(self.rem_outliers)])
        return string


def matching_from_string(string):
    """
    Creates a matching setup from a string.
    :param string:
    :param experiment: optional, only needed if k_max and rem_outliers are not given in the string.
    :return:
    """
    parts = [part.strip() for part in string.split("_")]

    space = None
    for s in Space:
        if s.value == parts[0]:
            space = s

    weighting = None
    for w in Weighting:
        if w.value == parts[1]:
            weighting = w

    assert space is not None and weighting is not None

    levels = int(parts[2])

    k_max = int(parts[3])
    rem_outliers = int(parts[4])

    return MatchingSetup(space, weighting, levels, k_max, rem_outliers)

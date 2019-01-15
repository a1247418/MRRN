from enum import Enum


class RegisteredLosses(Enum):
    MSE_F = "MSE_F"
    MSE_CF = "MSE_CF"
    REG = "REG"
    WASS = "WASS"


def needs_propensity(loss):
    """
    Checks whether a loss requires a propensity model.
    :param loss: loss string
    :return: True if propensity model is required, else False.
    """
    needs_prop = False
    if any([loss.startswith(string) for string in ["P_","KL_","PM_"]]):
        needs_prop = True

    return needs_prop
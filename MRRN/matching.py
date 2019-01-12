import os
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

import data_loader
from matching_defs import Group, Weighting


def _pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * np.matmul(X, np.transpose(Y))
    nx = np.sum(np.square(X), 1, keepdims=True)
    ny = np.sum(np.square(Y), 1, keepdims=True)
    D = (C + np.transpose(ny)) + nx
    return D


def _pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return np.sqrt(np.clip(_pdist2sq(X, Y), 1e-10, np.inf))


def produce_grouping(t, s, treatment_types_with_control, levels=1):
    """
    Preprocessing to matching: Groups members of different treatments or treatment-subranges into lists.
    Assumes s to be in range [0,1]
    :param t: Treatment vectors of dimension n_samples x n_treatments
    :param s: Treatment strength vectors of dimension n_samples x n_treatments
    :param levels: Into how many parts the treatment range should be partitioned.
    :return: Vector of group indicator. Lists of groupings: n_groups x n_grouping_members
    """
    assert levels >= 1

    n_treatments = len(s[0])

    group_list = []
    group_definitions = []
    for treatment in range(n_treatments):
        for level in range(levels):
            if treatment_types_with_control[treatment] == 1 or level == 0:
                group_list.append(np.array([]))
                group_id = len(group_list)-1

                s_t = s[t == treatment, treatment]
                idx_s_t = np.arange(len(t))[t==treatment]

                lower_bound = np.percentile(s_t, 100*level/levels) * (level > 0)
                upper_bound = np.percentile(s_t, 100*(level+1)/levels)
                group_definitions.append(Group(treatment, lower_bound, upper_bound))

                for i in range(len(s_t)):
                    if lower_bound < s_t[i] <= upper_bound:
                        group_list[group_id] = np.append(group_list[group_id], idx_s_t[i])

                group_list[group_id] = group_list[group_id].astype(int)

    return group_list, group_definitions



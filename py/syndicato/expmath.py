import numpy as np


def update_running_average(count, prev_average, new_value):
    """
    Updates an average computed on (count - 1) values with a new value
    :param count: Total count, post the addition of new_value
    :param prev_average: previous average value
    :param new_value: new value to update average with
    :return: new running average
    """
    return ((count - 1.0) * prev_average + new_value) / count


def upper_confidence_bound(t, n):
    """
    :param t: time step
    :param n: number of times arm has been selected
    :return:
    """

    return np.sqrt((2*np.log(t))/n)

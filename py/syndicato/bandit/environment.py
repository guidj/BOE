import abc

import numpy as np


class ContextFreeArm(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def draw(self):
        pass

    @property
    @abc.abstractproperty
    def name(self):
        pass


class BernoulliArm(ContextFreeArm):
    """
    Arm with Bernoulli distribution. Useful to simulate binary outcomes, e.g. click-through rate, with fixed
    probability.
    """
    def __init__(self, name, p, reward):
        assert isinstance(p, float)
        assert 0. <= p <= 1.0, "p should be between [0, 1]"
        assert isinstance(reward, float)
        assert reward >= 0, "reward should be a positive number"

        self._name = name
        self.p = p
        self.reward = reward

    def draw(self):
        """
        :return: returns 0 with probability `1-p` and `reward` with probability `p`.
        """
        if np.random.random() > self.p:
            return 0.0
        else:
            return self.reward

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "BernoulliArm(p={}, reward={}, name={})".format(self.p, self.reward, self.name)

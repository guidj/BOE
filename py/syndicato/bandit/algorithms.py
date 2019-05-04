"""
This module contains algorithms for context-free multi-armed bandits

Note that in most cases, internal state is only modified by the update function, which
can update the pull count, temperature, and rewards.
Updating counts, or parameters like temperature, without updating rewards leads to inconsistent and undesired behaviour.
"""

import abc

import numpy as np

from syndicato import expmath
from syndicato import logger


ANNEALING_FACTOR = 0.0000001
FLOATING_BOUND = 1e-8


class BanditAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    def update(self, chosen_arm, reward):
        pass


class ContextFreeAlgorithm(BanditAlgorithm):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def pull(self):
        pass

    def update(self, chosen_arm, reward):
        pass


class EpsilonGreedy(ContextFreeAlgorithm):
    """
    If exploration is set to 0 and no historical context is present, the algorithm will always pick the first option,
    since they each have equal payoff.

    If it's set to 1.0 (always explore), then random selection will occur on every pull.
    """

    def __init__(self, num_arms, epsilon):
        super(EpsilonGreedy, self).__init__()
        assert isinstance(num_arms, int)
        assert num_arms > 1, "there should be more than one arm"
        assert isinstance(epsilon, float)
        assert 0.0 <= epsilon <= 1.0, "epsilon should be between [0, 1]"
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.__counts = np.zeros(num_arms, dtype=np.uint32)
        self.__values = np.zeros(num_arms, dtype=np.float32)

    def pull(self):
        if np.random.random() > self.epsilon:
            # if the highest values are equal, they are equally likely to be selected
            highest_value = self.__values[np.argmax(self.__values)]
            highest_arm_indices = np.reshape(np.argwhere(self.__values == highest_value), (-1,))
            if highest_arm_indices.size == 1:
                return highest_arm_indices[0]
            else:
                # pick one of the best at random
                return highest_arm_indices[np.random.randint(0, highest_arm_indices.size)]
        else:
            # random selection
            return np.random.randint(0, self.num_arms)

    def update(self, chosen_arm, reward):
        self.__counts[chosen_arm] += 1
        prev_reward = self.__values[chosen_arm]

        self.__values[chosen_arm] = expmath.update_running_average(
            count=self.__counts[chosen_arm],
            prev_average=prev_reward,
            new_value=reward
        )

    @property
    def pull_counts(self):
        return self.__counts

    @property
    def rewards(self):
        return self.__values

    @staticmethod
    def create(epsilon, rewards, pulls=None):
        algorithm = EpsilonGreedy(num_arms=len(rewards), epsilon=epsilon)
        algorithm.__values = rewards
        if pulls:
            algorithm.__counts = pulls
        return algorithm


class Softmax(ContextFreeAlgorithm):
    def __init__(self, num_arms, temperature):
        assert isinstance(num_arms, int)
        assert num_arms > 1, "there should be more than one arm"
        assert isinstance(temperature, (int, float))
        assert temperature > 0, "temperature should positive"
        if temperature < 1e-2:
            logger.warning("Temperature [%f] is below 1e-2. Running risk arithmetic overflow, i.e. computational "
                           "frostbite.", temperature)
        self.num_arms = num_arms
        self.temperature = float(temperature)
        self.__counts = np.zeros(num_arms, dtype=np.uint32)
        self.__values = np.zeros(num_arms, dtype=np.float32)

    def pull(self):
        """
        TODO: Improve logic for handling inf and zero in exp
        """
        exp = np.exp(self.__values / self.temperature)
        p = exp / np.sum(exp)

        try:
            idx = np.random.choice(a=self.num_arms, p=p)
        except ValueError:
            inf_idx = np.isinf(p)
            inf, = np.where(inf_idx)
            if inf.size > 0:
                idx = np.random.randint(0, inf.size)
                return inf[idx]
            else:
                nan_idx = np.isnan(p)
                nan, = np.where(nan_idx)
                idx = np.random.randint(0, nan.size)
                return nan[idx]

        return idx

    def update(self, chosen_arm, reward):
        self.__counts[chosen_arm] += 1
        prev_reward = self.__values[chosen_arm]

        self.__values[chosen_arm] = expmath.update_running_average(
            count=self.__counts[chosen_arm],
            prev_average=prev_reward,
            new_value=reward
        )

    @property
    def pull_counts(self):
        return self.__counts

    @property
    def rewards(self):
        return self.__values

    @staticmethod
    def create(temperature, rewards, pulls=None):
        algorithm = Softmax(num_arms=len(rewards), temperature=temperature)
        algorithm.__values = rewards
        if pulls:
            algorithm.__counts = pulls
        return algorithm


class UCB1(ContextFreeAlgorithm):
    def __init__(self, num_arms, min_reward, max_reward):
        """
        For UCB1, rewards are assumed to be between 0 and 1
        Therefore, when computing the upper confidence bounds, rewards should be scaled between 0 and 1.
        The min_reward and reward_range parameters are used to scale the rewards.

        E.g. min-reward: -1, reward-range: 2
        reward t1: 0.7 -> (0.7 * 2) + (-1) = 0.4 in the scale scale of [0, 1]

        Note: if rewards are defined to be outside of the [0, 1] range, then the values
        fed into the update function should match the defined range.
        """
        assert num_arms > 1, "there should be more than one arm"
        assert max_reward > min_reward, "max-reward should be greater than min-reward"
        self.num_arms = num_arms
        self.min_reward = min_reward
        self.reward_range = max_reward - min_reward
        self.__counts = np.zeros(num_arms, dtype=np.uint32)
        self.__values = np.zeros(num_arms, dtype=np.float32)

    def pull(self):
        """
        Note: UCB1 tries to play every arm that hasn't been played first before optimizing.
        In a delayed reward scenario, we need to make sure all arms that haven't been played
        get played with equal probability. Otherwise, the same arm would be selected, and
        reward information would be available for it.
        :return:
        """
        # note: in a delayed reward context, there won't be any feedback about any arm
        # until the initial update
        unplayed, = np.where(self.__counts == 0)
        if unplayed.size > 0:
            return np.random.choice(a=unplayed)

        ucb = [0.0 for _ in range(self.num_arms)]
        pulls = np.sum(self.__counts)
        for arm in range(self.num_arms):
            bonus = np.sqrt((2.0 * np.log(pulls)) / float(self.__counts[arm]))
            scaled_value = (self.__values[arm] - self.min_reward) / self.reward_range
            ucb[arm] = scaled_value + bonus
        return np.argmax(ucb)

    def update(self, chosen_arm, reward):
        self.__counts[chosen_arm] += 1
        prev_reward = self.__values[chosen_arm]

        self.__values[chosen_arm] = expmath.update_running_average(
            count=self.__counts[chosen_arm],
            prev_average=prev_reward,
            new_value=reward
        )

    @property
    def pull_counts(self):
        return self.__counts

    @property
    def rewards(self):
        return self.__values

    @staticmethod
    def create(rewards, pulls=None, min_reward=0.0, max_reward=1.0):
        algorithm = UCB1(num_arms=len(rewards), min_reward=min_reward, max_reward=max_reward)
        algorithm.__values = rewards
        if pulls:
            algorithm.__counts = pulls
        return algorithm

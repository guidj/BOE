import abc

import numpy as np

from syndicato import expmath


class BanditAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    def update(self, chosen_arm, reward):
        pass


class ContextFreeAlgorithm(BanditAlgorithm):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_arm(self):
        pass

    def update(self, chosen_arm, reward):
        pass


class EpsilonGreedyAlgorithm(ContextFreeAlgorithm):
    """
    If exploration is set to 0 and no historical context is present, the algorithm will always pick the first option,
    since they each have equal payoff.

    If it's set to 1.0 (always explore), then random selection will occur on every pull.
    """

    def __init__(self, num_arms, epsilon):
        super(EpsilonGreedyAlgorithm, self).__init__()
        assert isinstance(num_arms, int)
        assert (num_arms > 0, "there should be at least one arm")
        assert isinstance(epsilon, float)
        assert (0.0 <= epsilon <= 1.0, "epislon should be between [0, 1]")
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms, dtype=np.int32)
        self.values = np.zeros(num_arms, dtype=np.float32)

    def select_arm(self):
        if np.random.random() > self.epsilon:
            # if the highest values are equal, they are equally likely to be selected
            highest_value = self.values[np.argmax(self.values)]
            highest_arm_indices = np.reshape(np.argwhere(self.values == highest_value), (-1,))
            if highest_arm_indices.size == 1:
                return highest_arm_indices[0]
            else:
                # pick one of the best at random
                return highest_arm_indices[np.random.randint(0, highest_arm_indices.size)]
        else:
            # random selection
            return np.random.randint(0, self.num_arms)

    def update(self, chosen_arm, reward):
        n = self.counts[chosen_arm] + 1
        prev_reward = self.values[chosen_arm]

        self.values[chosen_arm] = expmath.update_running_average(
            count=n,
            prev_average=prev_reward,
            new_value=reward
        )
        self.counts[chosen_arm] += 1

import unittest

from hypothesis import given
from hypothesis import assume
from hypothesis import settings
import hypothesis.strategies as st
import numpy as np

from syndicato.bandit.algorithms import EpsilonGreedy
from syndicato.bandit.algorithms import Softmax
from syndicato.bandit.algorithms import UCB1


class EspilonGreedyTest(unittest.TestCase):
    MAX_ARMS = 5000

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=10)
    def test_init_set_config_correctly(self, k, e):
        egreedy = EpsilonGreedy(num_arms=k, epsilon=e)

        self.assertEqual(k, egreedy.num_arms)
        self.assertEqual(e, egreedy.epsilon)

    @given(st.integers(max_value=1),
           st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=10)
    def test_init_fails_with_invalid_nums_arms(self, k, e):
        with self.assertRaises(AssertionError, msg="there should be more than one arm"):
            _ = EpsilonGreedy(num_arms=k, epsilon=e)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(max_value=0.0, exclude_max=True))
    @settings(max_examples=10)
    def test_init_fails_with_negative_epsilon(self, k, e):
        assume(e < 0.0)
        with self.assertRaises(AssertionError, msg="epsilon should be between [0, 1]"):
            _ = EpsilonGreedy(num_arms=k, epsilon=e)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=1.0, exclude_min=True))
    @settings(max_examples=10)
    def test_init_fails_with_epsilon_above_one(self, k, e):
        assume(e > 1.0)
        with self.assertRaises(AssertionError, msg="epsilon should be between [0, 1]"):
            _ = EpsilonGreedy(num_arms=k, epsilon=e)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=0.0, max_value=1.0),
           st.integers(min_value=1, max_value=1000))
    def test_pull_returns_a_valid_arm(self, k, e, rounds):
        egreedy = EpsilonGreedy(num_arms=k, epsilon=e)
        for _ in range(rounds):
            arm = egreedy.pull()

            self.assertGreaterEqual(arm, 0)
            self.assertLess(arm, k)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=0.0, max_value=1.0),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_pull_does_not_change_state(self, k, e, rounds):
        egreedy = EpsilonGreedy(num_arms=k, epsilon=e)

        for _ in range(rounds):
            egreedy.pull()

        self.assertEqual(0, sum(egreedy.pull_counts))
        self.assertEqual(0, sum(egreedy.rewards))

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=0.0, max_value=1.0),
           st.integers(min_value=1, max_value=1000),
           st.integers(1, 10))
    @settings(max_examples=10)
    def test_pull_does_not_change_state_after_any_updates(self, k, e, rounds, updates):
        egreedy = EpsilonGreedy(num_arms=k, epsilon=e)

        for _ in range(updates):
            egreedy.update(chosen_arm=np.random.randint(0, k), reward=np.random.random())

        pulls = sum(egreedy.pull_counts)
        rewards = sum(egreedy.rewards)

        for _ in range(rounds):
            egreedy.pull()

        self.assertEqual(pulls, sum(egreedy.pull_counts))
        self.assertEqual(rewards, sum(egreedy.rewards))

    @given(st.integers(min_value=2, max_value=5),
           st.floats(min_value=0.0, max_value=0.8, exclude_max=True),
           st.integers(min_value=1000, max_value=5000))
    @settings(deadline=500, max_examples=10)
    def test_the_best_arm_based_on_feedback(self, k, e, rounds):
        p = np.random.random(k)
        p = p / np.sum(p)
        egreedy = EpsilonGreedy.create(e, rewards=p)
        pulls = np.zeros(k, dtype=np.uint64)

        for _ in range(rounds):
            arm = egreedy.pull()
            pulls[arm] += 1

        self.assertEqual(np.argmax(p), np.argmax(pulls))


class SoftmaxTest(unittest.TestCase):
    MAX_ARMS = 5000
    MAX_TEMP = 10000

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=1e-2, max_value=MAX_TEMP, exclude_min=True))
    @settings(max_examples=10)
    def test_init_set_config_correctly(self, k, tau):
        softmax = Softmax(num_arms=k, temperature=tau)

        self.assertEqual(k, softmax.num_arms)
        self.assertEqual(tau, softmax.temperature)

    @given(st.integers(max_value=1),
           st.floats(min_value=1e-2, max_value=MAX_TEMP, exclude_min=True))
    @settings(max_examples=10)
    def test_init_fails_with_invalid_nums_arms(self, k, tau):
        with self.assertRaises(AssertionError, msg="there should be more than one arm"):
            _ = Softmax(num_arms=k, temperature=tau)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(max_value=0.0, exclude_max=True))
    @settings(max_examples=10)
    def test_init_fails_with_negative_temperature(self, k, tau):
        assume(tau < 0.0)
        with self.assertRaises(AssertionError, msg="temperature should positive"):
            _ = Softmax(num_arms=k, temperature=tau)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=1e-2, max_value=MAX_TEMP, exclude_min=True),
           st.integers(min_value=1, max_value=1000))
    @settings(deadline=1000)  # can be slow with the current implementation
    def test_pull_returns_a_valid_arm(self, k, tau, rounds):
        softmax = Softmax(num_arms=k, temperature=tau)

        for _ in range(rounds):
            arm = softmax.pull()
            self.assertGreaterEqual(arm, 0)
            self.assertLess(arm, k)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=1e-2, max_value=MAX_TEMP, exclude_min=True),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_pull_does_not_change_state(self, k, tau, rounds):
        softmax = Softmax(num_arms=k, temperature=tau)

        for _ in range(rounds):
            softmax.pull()

        self.assertEqual(0, sum(softmax.pull_counts))
        self.assertEqual(0, sum(softmax.rewards))

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(min_value=1e-2, max_value=MAX_TEMP),
           st.integers(min_value=1, max_value=1000),
           st.integers(1, 10))
    def test_pull_does_not_change_state_after_any_updates(self, k, tau, rounds, updates):
        softmax = Softmax(num_arms=k, temperature=tau)

        for _ in range(updates):
            softmax.update(chosen_arm=np.random.randint(0, k), reward=np.random.random())

        pulls = sum(softmax.pull_counts)
        rewards = sum(softmax.rewards)

        for _ in range(rounds):
            softmax.pull()

        self.assertEqual(pulls, sum(softmax.pull_counts))
        self.assertEqual(rewards, sum(softmax.rewards))

    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=5000, max_value=10000))
    @settings(deadline=500, max_examples=10)
    def test_the_best_arm_based_on_feedback(self, k, rounds):
        p = np.random.random(k)
        p = p / np.sum(p)
        softmax = Softmax.create(temperature=1.0, rewards=p)
        pulls = np.zeros(k, dtype=np.uint64)

        for _ in range(rounds):
            arm = softmax.pull()
            pulls[arm] += 1

        self.assertEqual(np.argmax(p), np.argmax(pulls))


class UCB1Test(unittest.TestCase):
    MAX_ARMS = 5000

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(),
           st.floats())
    @settings(max_examples=10)
    def test_init_set_config_correctly(self, k, min_reward, max_reward):
        assume(min_reward < max_reward)
        ucb1 = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

        self.assertEqual(k, ucb1.num_arms)
        self.assertEqual(min_reward, ucb1.min_reward)
        self.assertEqual(max_reward - min_reward, ucb1.reward_range)

    @given(st.integers(max_value=1),
           st.floats(),
           st.floats())
    @settings(max_examples=10)
    def test_init_fails_with_invalid_nums_arms(self, k, min_reward, max_reward):
        with self.assertRaises(AssertionError, msg="there should be more than one arm"):
            _ = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(),
           st.floats())
    @settings(max_examples=10)
    def test_init_fails_with_invalid_reward_bounds(self, k, min_reward, max_reward):
        assume(min_reward >= max_reward)
        with self.assertRaises(AssertionError, msg="max-reward should be greater than min-reward"):
            _ = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(),
           st.floats(),
           st.integers(min_value=1, max_value=1000))
    def test_pull_returns_a_valid_arm(self, k, min_reward, max_reward, rounds):
        assume(min_reward < max_reward)
        ucb1 = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

        for _ in range(rounds):
            arm = ucb1.pull()
            self.assertGreaterEqual(arm, 0)
            self.assertLess(arm, k)

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(),
           st.floats(),
           st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_pull_does_not_change_state(self, k, min_reward, max_reward, rounds):
        assume(min_reward < max_reward)
        ucb1 = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

        for _ in range(rounds):
            ucb1.pull()

        self.assertEqual(0, sum(ucb1.pull_counts))
        self.assertEqual(0, sum(ucb1.rewards))

    @given(st.integers(min_value=2, max_value=MAX_ARMS),
           st.floats(),
           st.floats(),
           st.integers(min_value=1, max_value=1000),
           st.integers(1, 10))
    def test_pull_does_not_change_state_after_any_updates(self, k, min_reward, max_reward, rounds, updates):
        assume(min_reward < max_reward)
        ucb1 = UCB1(num_arms=k, min_reward=min_reward, max_reward=max_reward)

        for _ in range(updates):
            ucb1.update(chosen_arm=np.random.randint(0, k), reward=np.random.random())

        pulls = sum(ucb1.pull_counts)
        rewards = sum(ucb1.rewards)

        for _ in range(rounds):
            ucb1.pull()

        self.assertEqual(pulls, sum(ucb1.pull_counts))
        self.assertEqual(rewards, sum(ucb1.rewards))

    @given(st.integers(min_value=2, max_value=5),
           st.floats(min_value=-100, max_value=100),
           st.floats(min_value=-100, max_value=100),
           st.integers(min_value=1000, max_value=5000))
    @settings(deadline=500, max_examples=10)
    def test_the_best_arm_based_on_feedback(self, k, min_reward, max_reward, rounds):
        assume(min_reward < max_reward)
        rrange = max_reward - min_reward
        p = np.random.random(k)
        p = p / np.sum(p)
        # need to re-scale rewards to match defined range
        p = p * rrange
        # we need to set pulls; otherwise, arms will be played to maximize learning
        ucb1 = UCB1.create(rewards=p, pulls=[rounds] * p.size, min_reward=min_reward, max_reward=max_reward)
        pulls = np.zeros(k, dtype=np.uint64)

        for _ in range(rounds):
            arm = ucb1.pull()
            pulls[arm] += 1

        self.assertEqual(np.argmax(p), np.argmax(pulls))


if __name__ == '__main__':
    unittest.main()

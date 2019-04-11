import numpy as np

from syndicato import expmath


class SimulationResult(object):
    def __init__(self, num_arms, num_sims, num_trials, chosen_arms, rewards, cumulative_rewards):
        """
        Results of a simulation. Each simulation will have several runs, each with a fixed number of trials.
        For example, a simulation can run 20 times and each time it runs there can be 100 trials.
        This gives a total of 2000 trials. Each input is a 2D matrix representing this
        :param chosen_arms: matrix of arms chosen in each trial
        :param rewards: matrix of rewards observed in each trial
        :param cumulative_rewards: matrix of cumulative rewards observed after each trial
        """
        assert chosen_arms.size == (num_sims * num_trials), "chosen_arms does match simulations x trials"
        assert rewards.size == (num_sims * num_trials), "rewards does match simulations x trials"
        assert cumulative_rewards.size == (num_sims * num_trials), "cumulative_rewards does match simulations x trials"
        self.num_arms = num_arms
        self.num_trials = num_trials
        self.num_sims = num_sims
        self.chosen_arms = chosen_arms
        self.rewards = rewards
        self.cumulative_rewards = cumulative_rewards


class SimulationStats(object):
    def __init__(self, simulation_result):
        """
        Produces grids for rewards, cumulative rewards, and selection probability:
            # arm-1: [0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 4, 5, 5, 6, ....]
            # arm-2: [0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 4, 5, 5, 6, ....]
            # arm-3: [0, 1, 3, 4, 345, 6, ...]


        Average reward is only updated if the arm is selected
        Probability of selection is updated for each arm in each timestep
        As is cumulative reward. But since it's just addition, it's always zero.
        :param simulation_result: a SimulationResult object
        """
        assert isinstance(simulation_result, SimulationResult)
        self.selection_probabilities = np.zeros([simulation_result.num_arms, simulation_result.num_trials],
                                                dtype=np.float32)
        self.average_rewards = np.zeros([simulation_result.num_arms, simulation_result.num_trials], dtype=np.float32)
        self.average_rewards_ci = np.zeros([simulation_result.num_arms, simulation_result.num_trials], dtype=np.float32)
        self.cumulative_rewards = np.zeros([simulation_result.num_arms, simulation_result.num_trials], dtype=np.float32)

        selection_counts = None
        avg_rewards = None

        # average results per arm, over different runs
        for sim in range(simulation_result.num_sims):
            for trial in range(simulation_result.num_trials):
                selected_arm = simulation_result.chosen_arms[sim, trial]

                if trial == 0:
                    selection_counts = np.zeros(simulation_result.num_arms, dtype=np.int32)
                    avg_rewards = np.zeros(simulation_result.num_arms, dtype=np.float32)

                # selection prob, cumulative reward, and ci should be update for every arm on each time step
                selection_counts[selected_arm] += 1
                for arm in range(simulation_result.num_arms):
                    prob_select = selection_counts[arm] / np.float32(trial + 1)

                    if sim == 0:
                        prev_prob_select = prob_select
                    else:
                        prev_prob_select = self.selection_probabilities[arm, trial]

                    self.selection_probabilities[arm, trial] = \
                        expmath.update_running_average(count=sim + 1,
                                                       prev_average=prev_prob_select,
                                                       new_value=prob_select)
                    if trial > 0:
                        self.cumulative_rewards[arm, trial] = self.cumulative_rewards[arm, trial - 1]

                    if arm == selected_arm:
                        self.cumulative_rewards[arm, trial] += simulation_result.rewards[sim, trial]

                    if selection_counts[arm] > 0:
                        self.average_rewards_ci[arm, trial] = expmath.upper_confidence_bound(trial + 1,
                                                                                             selection_counts[arm])

                # update average reward / only applies to selected arm
                if sim == 0:
                    prev_avg_reward = simulation_result.rewards[sim, trial]

                else:
                    prev_avg_reward = self.average_rewards[selected_arm, trial]

                avg_rewards[selected_arm] = expmath.update_running_average(count=trial + 1,
                                                                           prev_average=avg_rewards[selected_arm],
                                                                           new_value=simulation_result.rewards[
                                                                               sim, trial])

                self.average_rewards[selected_arm, trial] = \
                    expmath.update_running_average(count=sim + 1,
                                                   prev_average=prev_avg_reward,
                                                   new_value=avg_rewards[selected_arm])

    def __repr__(self):
        def pprint(grid):
            return '\n'.join(['\t'.join(['{:2f}'.format(cell) for cell in row]) for row in grid])

        return "Rewards\n{}\n\nCumulative Rewards\n{}\n\nSelection Probability\n{}" \
            .format(pprint(self.average_rewards), pprint(self.cumulative_rewards), pprint(self.selection_probabilities))


class ContextFreeBanditSimulation(object):
    """
    Runs num_sims simulations with an algorithm returned by `algorithm_fn`.
    On each simulation run, a new instance of the algorithm is generated by calling the `algorithm_fn`.
    Then, num_trials ensue, where on each trial, an arm is selected, it's reward retrieved, and the algorithm updated.

    Note that this does not currently support delayed reward
    """

    @staticmethod
    def run(algorithm_fn, arms, num_sims, num_trials):
        chosen_arms = np.zeros([num_sims, num_trials], dtype=np.int32)
        rewards = np.zeros([num_sims, num_trials], dtype=np.float32)
        cumulative_rewards = np.zeros([num_sims, num_trials], dtype=np.float32)

        for sim in range(num_sims):
            algorithm = algorithm_fn()

            for trial in range(num_trials):
                trial = trial

                chosen_arm = algorithm.select_arm()
                chosen_arms[sim, trial] = chosen_arm
                reward = arms[chosen_arm].draw()
                rewards[sim, trial] = reward

                if trial == 1:
                    cumulative_rewards[sim, trial] = reward
                else:
                    cumulative_rewards[sim, trial] = cumulative_rewards[sim, trial - 1] + reward

                algorithm.update(chosen_arm, reward)

        return SimulationResult(
            num_arms=len(arms),
            num_sims=num_sims,
            num_trials=num_trials,
            chosen_arms=chosen_arms,
            rewards=rewards,
            cumulative_rewards=cumulative_rewards
        )

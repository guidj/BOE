class ArmSnapshot(object):
    def __init__(self, prob_selected, avg_reward, total_reward, ucb):
        self.snapshot = [prob_selected, avg_reward, total_reward, ucb]

    @property
    def prob_selected(self):
        return self.snapshot[0]

    @property
    def avg_reward(self):
        return self.snapshot[1]

    @property
    def total_reward(self):
        return self.snapshot[2]

    @property
    def ucb(self):
        return self.snapshot[3]


class SimulationResult(object):
    def __init__(self, algorithm_final_state, num_arms, num_trials, step_snapshots):
        """
        Results of a simulation. Each simulation will be a single run, with several trials
        arms_snapshots: list of list with ArmSnapshot instances
        """
        self.algorithm_final_state = algorithm_final_state
        self.num_arms = num_arms
        self.num_trials = num_trials
        self.step_snapshots = step_snapshots


class ExperimentParams(object):
    def __init__(self, num_sims, num_trials, update_delay, update_steps, snapshot_steps):
        self.num_sims = num_sims
        self.num_trials = num_trials
        self.update_delay = update_delay
        self.update_steps = update_steps
        self.snapshot_steps = snapshot_steps


class ExperimentStats(object):
    def __init__(self, num_sims, arms_prob_selection, arms_average_rewards, arms_cumulative_rewards, arms_ucb):
        self.num_sims = num_sims
        self.arms_prob_selection = arms_prob_selection
        self.arms_average_rewards = arms_average_rewards
        self.arms_cumulative_rewards = arms_cumulative_rewards
        self.arms_ucb = arms_ucb

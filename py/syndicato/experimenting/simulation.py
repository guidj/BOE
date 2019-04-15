import numpy as np

from syndicato import expmath
from syndicato.experimenting import record


def arm_snapshot_from_state(step, times_selected, avg_reward, total_reward, ucb=None):
    return record.ArmSnapshot(
        prob_selected=times_selected / float(step),
        avg_reward=avg_reward,
        total_reward=total_reward,
        ucb=ucb
    )


def exp_stats_from_simulation_results(simulation_results):
    assert isinstance(simulation_results, (list, tuple, set))

    values = []
    for sr in simulation_results:
        assert isinstance(sr, record.SimulationResult)
        matrix = []
        for arms_snapshots in sr.step_snapshots:
            matrix.append([arm_snapshot.snapshot for arm_snapshot in arms_snapshots])
        values.append(matrix)

    # (simulation, step, arm, value) -> (step, arm, value)
    results = np.average(np.array(values, dtype=np.float32), axis=0)

    # prob_selected, avg_reward, total_reward, ucb
    arms_prob_selection = results[:, :, 0]
    arms_average_rewards = results[:, :, 1]
    arms_cumulative_rewards = results[:, :, 2]
    arms_ucb = results[:, :, 3]

    return record.ExperimentStats(
        num_sims=len(simulation_results),
        arms_prob_selection=arms_prob_selection,
        arms_average_rewards=arms_average_rewards,
        arms_cumulative_rewards=arms_cumulative_rewards,
        arms_ucb=arms_ucb
    )


class DelayedCounter(object):
    def __init__(self, starting_step):
        self.starting_step = starting_step
        self.__c = 0

    def inc(self, step, by=None):
        if step >= self.starting_step:
            self.__c += by if by is not None else 1

    @property
    def value(self):
        return self.__c


class ContextFreeBanditSimulator(object):
    @staticmethod
    def run(algorithm, arms, exp_params):
        return ContextFreeBanditSimulator.__simulate(algorithm, arms, exp_params)

    @staticmethod
    def __simulate(algorithm, arms, exp_params):
        step_snapshots = []

        chosen_arms = []
        total_rewards = [0] * len(arms)
        update_counter = DelayedCounter(exp_params.update_delay)
        for trial in range(exp_params.num_trials):
            step = trial + 1
            update_counter.inc(step)
            chosen_arm = algorithm.pull()
            chosen_arms.append(chosen_arm)

            if step >= exp_params.update_delay and update_counter.value % exp_params.update_steps == 0:
                new_total_rewards = ContextFreeBanditSimulator.__update(algorithm, arms, chosen_arms)
                total_rewards = np.sum([total_rewards, new_total_rewards], axis=0)
                chosen_arms = []

            if step % exp_params.snapshot_steps == 0:
                # Note: this only snapshots what the algorithm knows, not the current state
                # E.g. An algorithm runs daily, feedback is known after two days, but the algorithm is only updated
                # every 10 days. A snapshot on day 7 would reflect the rewards known to the algorithm when it
                # was initialized, but the known performance would cover actions taken on days 1 to 5.
                step_snapshot = [
                    arm_snapshot_from_state(step=step,
                                            times_selected=algorithm.pull_counts[idx],
                                            avg_reward=algorithm.rewards[idx],
                                            total_reward=total_rewards[idx],
                                            ucb=expmath.upper_confidence_bound(t=step,
                                                                               n=algorithm.pull_counts[idx])) for idx in
                    range(len(arms))
                ]
                step_snapshots.append(step_snapshot)

        return record.SimulationResult(
            algorithm_final_state=algorithm,
            num_arms=len(arms),
            num_trials=exp_params.num_trials,
            step_snapshots=step_snapshots
        )

    @staticmethod
    def __update(algorithm, arms, chosen_arms):
        total_rewards = [0.0] * len(arms)
        for chosen_arm in chosen_arms:
            reward = arms[chosen_arm].draw()
            total_rewards[chosen_arm] += reward
            algorithm.update(chosen_arm=chosen_arm, reward=reward)

        return total_rewards

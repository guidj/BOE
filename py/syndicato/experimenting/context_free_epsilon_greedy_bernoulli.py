import argparse
import csv
import datetime
import os
import os.path

import numpy as np

from syndicato import logger
from syndicato.bandit import algorithms
from syndicato.bandit import environment
from syndicato.bandit import simulation
from syndicato.reporting import plotting
from syndicato.reporting import visualization


def main(report_path, logs_path):
    if os.path.exists(logs_path):
        os.remove(logs_path)

    names = ['jam', 'box', 'dw', 'dm', 'sc', 'ds']
    means = [0.04, 0.03, 0.02, 0.01, 0.01, 0.033]
    np.random.shuffle(means)
    num_arms = len(means)
    arms = [environment.BernoulliArm(p=mean, reward=1.0, name=name) for name, mean in zip(names, means)]

    def create_algorithm_fn(epsilon, num_arms):
        def algorithm_fn():
            return algorithms.EpsilonGreedyAlgorithm(num_arms, epsilon)

        return algorithm_fn

    def create_row_generator_fn(simulation_result, num_sims, num_trials, epsilon):
        def fn():
            for sim in range(num_sims):
                for trial in range(num_trials):
                    row = [
                        epsilon,
                        sim + 1,
                        trial + 1,
                        simulation_result.chosen_arms[sim, trial],
                        simulation_result.rewards[sim, trial],
                        simulation_result.cumulative_rewards[sim, trial]
                    ]
                    yield row

        return fn

    logger.info('Arms:')
    for arm in arms:
        logger.info('\t%s', arm)

    results = []

    logger.info('Starting simulation')
    num_sims = 100
    num_trials = 10000
    for epsilon in [0.1]:
        exp_config = {
            'algorithm': 'epsilon-greedy',
            'epsilon': epsilon,
            'num-simulations': num_sims,
            'num-trials': num_trials
        }

        start = datetime.datetime.now()
        logger.info('Running with config: %s', exp_config)

        algorithm_fn = create_algorithm_fn(epsilon, num_arms)
        simulation_result = simulation.ContextFreeBanditSimulation.run(algorithm_fn,
                                                                       arms=arms,
                                                                       num_sims=num_sims,
                                                                       num_trials=num_trials)
        end = datetime.datetime.now()
        logger.info('Simulation ended after %s seconds]', (end - start).seconds)
        logger.info('Logging results %s' % logs_path)

        simulation_stats = simulation.SimulationStats(simulation_result)
        export_result_to_file(create_row_generator_fn(simulation_result, num_sims, num_arms, epsilon),
                              file_path=logs_path,
                              mode='wa')
        plots = generate_summary_plots(arms, simulation_stats)
        results.append((exp_config, plots))

    logger.info('Exporting results to %s', report_path)
    visualization.html_report(arms, results=results, output_path=report_path)


def export_result_to_file(row_generator, file_path, mode):
    with open(file_path, mode) as fp:
        writer = csv.writer(fp)
        for row in row_generator():
            writer.writerow(row)


def generate_summary_plots(arms, simulation_stats):
    import matplotlib.pyplot as plt

    parent_figure, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 12))
    parent_figure.subplots_adjust(wspace=0.35)

    plotting.generate_metric_figure_from_matrix(simulation_stats.selection_probabilities,
                                                arms=arms,
                                                ax=axes[0, 0],
                                                title='Selection Probability/Arm'),
    plotting.generate_metric_figure_from_matrix(simulation_stats.average_rewards,
                                                ci_matrix=simulation_stats.average_rewards_ci,
                                                arms=arms,
                                                ax=axes[0, 1],
                                                title='Average Reward/Arm'),
    plotting.generate_metric_figure_from_matrix(simulation_stats.cumulative_rewards,
                                                arms=arms,
                                                ax=axes[1, 0],
                                                title='Cumulative Reward/Arm'),
    plotting.generate_metric_figure_from_values(np.average(simulation_stats.cumulative_rewards, axis=0),
                                                label='cumulative-reward/global',
                                                ax=axes[1, 1],
                                                title='Cumulative Reward/Global')
    return [parent_figure]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Context-free bandit simulation')
    arg_parser.add_argument('--report-path', help='Path to save html report', required=True)
    arg_parser.add_argument('--logs-path', help='Path to save csv report', required=True)

    args = arg_parser.parse_args()
    main(args.report_path, args.logs_path)

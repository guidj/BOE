import argparse
import csv
import os
import os.path
import multiprocessing
import datetime
import shutil

import numpy as np

from syndicato import logger
from syndicato.bandit import algorithms
from syndicato.bandit import environment
from syndicato.experimenting import simulation
from syndicato.experimenting import record
from syndicato.reporting import plotting
from syndicato.reporting import visualization


def run_simulation(args):
    idx, algorithm, arms, exp_params = args
    start = datetime.datetime.now()
    simulation_result = simulation.ContextFreeBanditSimulator.run(algorithm=algorithm,
                                                                  arms=arms,
                                                                  exp_params=exp_params)
    end = datetime.datetime.now()
    logger.info('Simulation [%s] ran in %s seconds', idx, (end - start).seconds)
    return simulation_result


def prepare_report_export_dir(job_dir, params):
    tokens = []
    for k in sorted(params.keys()):
        tokens.append('%s-%s' % (k, params[k]))

    path_name = os.path.join(job_dir, '_'.join(tokens))

    if os.path.exists(path_name):
        if os.path.exists(path_name):
            logger.info('Found existing dir %s', path_name)
            shutil.rmtree(path_name)
            logger.info('Deleted existing dir %s', path_name)

    os.makedirs(path_name)

    return os.path.join(path_name, 'report.html')


def main(exp_num_sims, exp_num_trials, exp_update_delay, exp_update_steps, exp_snapshot_steps, job_dir, report_ci,
         report_ci_scaling_factor):
    names = ['jam', 'box', 'dw', 'dm', 'sc', 'ds']
    means = [0.02, 0.03, 0.02, 0.01, 0.01, 0.023]
    np.random.shuffle(means)
    num_arms = len(means)
    arms = [environment.BernoulliArm(p=mean, reward=1.0, name=name) for name, mean in zip(names, means)]
    exp_params = record.ExperimentParams(num_sims=exp_num_sims,
                                         num_trials=exp_num_trials,
                                         update_delay=exp_update_delay,
                                         update_steps=exp_update_steps,
                                         snapshot_steps=exp_snapshot_steps)

    def create_algorithm_fn(num_arms, epsilon):
        def algorithm_fn():
            return algorithms.EpsilonGreedyAlgorithm(num_arms=num_arms, epsilon=epsilon)

        return algorithm_fn

    logger.info('Arms:')
    for arm in arms:
        logger.info('\t%s', arm)

    results = []

    for epsilon in [0.1]:
        exp_config = {
            'algorithm': 'epsilon-greedy',
            'epsilon': epsilon,
            'num-simulations': exp_params.num_sims,
            'num-trials': exp_params.num_trials
        }

        report_path = prepare_report_export_dir(job_dir, params=exp_config)

        logger.info('Running with config: %s', exp_config)

        algorithm_fn = create_algorithm_fn(num_arms, epsilon)

        args = [(idx + 1, algorithm_fn(), arms, exp_params) for idx in range(exp_params.num_sims)]
        with multiprocessing.Pool() as mp:
            start = datetime.datetime.now()
            simulation_results = mp.map(func=run_simulation, iterable=args)
            end = datetime.datetime.now()
            logger.info('Experiment ran for %s seconds]', (end - start).seconds)

        experiment_result = simulation.exp_stats_from_simulation_results(simulation_results)

        logger.info('Experiment ended')

        plots = generate_summary_plots(arms, exp_params, experiment_result, report_ci, report_ci_scaling_factor)
        results.append((exp_config, plots))

    logger.info('Exporting results to %s', report_path)
    visualization.html_report(arms, results=results, output_path=report_path)


def export_result_to_file(row_generator, file_path, mode):
    with open(file_path, mode) as fp:
        writer = csv.writer(fp)
        for row in row_generator():
            writer.writerow(row)


def generate_summary_plots(arms, exp_params, experiment_stats, report_ci, ci_scaling_factor):
    assert isinstance(exp_params, record.ExperimentParams)
    assert isinstance(experiment_stats, record.ExperimentStats)
    import matplotlib.pyplot as plt

    parent_figure, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 12))
    parent_figure.subplots_adjust(wspace=0.35)

    steps = [step for step in range(1, exp_params.num_trials + 1, exp_params.snapshot_steps)]
    reward_ci_matrix = experiment_stats.arms_ucb * ci_scaling_factor if report_ci else None

    plotting.generate_metric_figure_from_matrix(steps,
                                                matrix=experiment_stats.arms_prob_selection,
                                                arms=arms,
                                                ax=axes[0, 0],
                                                title='Selection Probability/Arm'),
    plotting.generate_metric_figure_from_matrix(steps,
                                                matrix=experiment_stats.arms_average_rewards,
                                                ci_matrix=reward_ci_matrix,
                                                arms=arms,
                                                ax=axes[0, 1],
                                                title='Average Reward/Arm'),
    plotting.generate_metric_figure_from_matrix(steps,
                                                matrix=experiment_stats.arms_cumulative_rewards,
                                                arms=arms,
                                                ax=axes[1, 0],
                                                title='Cumulative Reward/Arm'),
    plotting.generate_metric_figure_from_values(steps,
                                                values=np.sum(experiment_stats.arms_cumulative_rewards, axis=1),
                                                label='cumulative-reward/global',
                                                ax=axes[1, 1],
                                                title='Cumulative Reward/Global')
    return [parent_figure]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('Context-free bandit simulation')
    arg_parser.add_argument('--job-dir', help='Path to save experiment report and logs', required=True)
    arg_parser.add_argument('--report-ci', help='Measure confidence interval (CI)', required=False, type=bool,
                            default=True)
    arg_parser.add_argument('--report-ci-scaling-factor', help='Scaling factor for CI', required=False, type=float,
                            default=0.01)
    arg_parser.add_argument('--exp-num-sims', help='Number of simulations', required=False, type=int,
                            default=50)
    arg_parser.add_argument('--exp-num-trials', help='Number of trials per simulations', required=False, type=int,
                            default=1000)
    arg_parser.add_argument('--exp-update-delay', help='Number of initial steps before the first update',
                            required=False, type=int,
                            default=0)
    arg_parser.add_argument('--exp-update-steps', help='Number of steps between updates. 1 means update on every steps',
                            required=False, type=int,
                            default=10)
    arg_parser.add_argument('--exp-snapshot-steps', help='Number of steps between snapshots', required=False, type=int,
                            default=10)

    args = arg_parser.parse_args()

    for arg in vars(args):
        logger.info('\t- %s: %s', arg, getattr(args, arg))

    main(args.exp_num_sims, args.exp_num_trials, args.exp_update_delay, args.exp_update_steps, args.exp_snapshot_steps,
         args.job_dir, args.report_ci, args.report_ci_scaling_factor)

import argparse
import csv
import os
import os.path
import multiprocessing
import datetime
import time
import shutil
import json
import hashlib

import numpy as np

from syndicato import logger
from syndicato.bandit import algorithms
from syndicato.bandit import environment
from syndicato.experimenting import simulation
from syndicato.experimenting import record
from syndicato.experimenting import config
from syndicato.reporting import plotting
from syndicato.reporting import visualization
from syndicato.reporting import persistence


def run_simulation(args):
    idx, algorithm, arms, exp_params = args
    start = datetime.datetime.now()
    simulation_result = simulation.ContextFreeBanditSimulator.run(algorithm=algorithm,
                                                                  arms=arms,
                                                                  exp_params=exp_params)
    end = datetime.datetime.now()
    logger.info('Simulation [%s] ran in %s seconds', idx, (end - start).seconds)
    return simulation_result


def prepare_export_dir(save_dir, params):
    tokens = []
    for k in sorted(params.keys()):
        tokens.append('%s-%s' % (k, params[k]))

    path_name = os.path.join(save_dir, '_'.join(tokens))

    if os.path.exists(path_name):
        if os.path.exists(path_name):
            logger.info('Found existing dir %s', path_name)
            shutil.rmtree(path_name)
            logger.info('Deleted existing dir %s', path_name)

    os.makedirs(path_name)

    return path_name


def create_arms(arms_config):
    arms = []
    for arm_config in arms_config:
        if arm_config['type'] == 'bernoulli':
            arm = environment.BernoulliArm(name=arm_config['name'], p=arm_config['p'], reward=arm_config['reward'])
            arms.append(arm)
        else:
            raise RuntimeError('Unknown arm type: %s', arm_config['type'])

    return arms


def create_algorithm_fn(typ, algorithm_config, num_arms):
    def algorithm_fn():
        if typ == 'epsilon-greedy':
            return algorithms.EpsilonGreedyAlgorithm(num_arms=num_arms, epsilon=algorithm_config['epsilon'])
        else:
            raise RuntimeError('Unknown algorithm: %s' % algorithm_config['id'])

    return algorithm_fn


def create_id_from_config(config_json):
    payload = json.dumps(config_json, sort_keys=True).encode('utf-8')
    return '%s_%d' % (hashlib.md5(payload).hexdigest()[:8], int(time.time()))


def main(config_file, job_dir, nproc):
    experiments_config = config.parse_config(path=config_file)
    experiment_results = []

    for exp_config in experiments_config['experiments']:
        experiment_id = '%s_%s' % (exp_config['id'], create_id_from_config(exp_config))
        experiment_dir = os.path.join(job_dir, experiment_id)
        report_config = exp_config['config']['report']
        arms = create_arms(exp_config['arms'])
        num_arms = len(arms)
        exp_params = record.ExperimentParams(num_sims=exp_config['config']['simulation']['runs'],
                                             num_trials=exp_config['config']['simulation']['trials'],
                                             update_delay=exp_config['config']['simulation']['update-delay'],
                                             update_steps=exp_config['config']['simulation']['update-steps'],
                                             snapshot_steps=exp_config['config']['simulation']['snapshot-steps'])
        logger.info('Arms:')
        for arm in arms:
            logger.info('\t%s', arm)

        for algorithm_config in exp_config['algorithm']['configs']:
            experiment_config_id = create_id_from_config(algorithm_config)
            params = {**{
                'algorithm': exp_config['algorithm']['id'],
                'runs': exp_params.num_sims,
                'trials': exp_params.num_trials,
                'id': experiment_config_id
            }, **algorithm_config}

            export_path = prepare_export_dir(experiment_dir, params=params)
            logger.info('Running with config: %s', params)

            algorithm_fn = create_algorithm_fn(typ=exp_config['algorithm']['id'],
                                               algorithm_config=algorithm_config,
                                               num_arms=num_arms)

            args = [(idx + 1, algorithm_fn(), arms, exp_params) for idx in range(exp_params.num_sims)]
            with multiprocessing.Pool(nproc) as mp:
                start = datetime.datetime.now()
                simulation_results = mp.map(func=run_simulation, iterable=args)
                end = datetime.datetime.now()
                logger.info('Experiment ran for %s seconds]', (end - start).seconds)

            experiment_stats = simulation.experiment_stats_from_simulation_results(simulation_results)

            logger.info('Experiment %s ended', exp_config['id'])
            persistence.export(export_path, experiment_stats)
            plots = generate_summary_plots(arms,
                                           exp_params=exp_params,
                                           experiment_stats=experiment_stats,
                                           ci_scaling_factor=report_config.get('ci-scaling-factor', None))
            visualization.html_report(arms, results=[(params, plots)],
                                      output_path=os.path.join(export_path, 'report.html'))
            experiment_results.append((params, plots))

            logger.info('Exporting results to %s', export_path)

        report_path = os.path.join(experiment_dir, 'report.html')
        visualization.html_report(arms, results=experiment_results, output_path=report_path)


def export_result_to_file(row_generator, file_path, mode):
    with open(file_path, mode) as fp:
        writer = csv.writer(fp)
        for row in row_generator():
            writer.writerow(row)


def generate_summary_plots(arms, exp_params, experiment_stats, ci_scaling_factor=None):
    assert isinstance(exp_params, record.ExperimentParams)
    assert isinstance(experiment_stats, record.ExperimentStats)
    import matplotlib.pyplot as plt

    parent_figure, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 12))
    parent_figure.subplots_adjust(wspace=0.35)

    steps = [step for step in range(1, exp_params.num_trials + 1, exp_params.snapshot_steps)]
    reward_ci_matrix = experiment_stats.arms_ucb * ci_scaling_factor if ci_scaling_factor else None

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
    arg_parser.add_argument('--config-file', help='Path to experiment config file', required=True)
    arg_parser.add_argument('--nproc', help='Sets concurrency level for simulation runs on each experiment',
                            required=False, type=int, default=1)

    args = arg_parser.parse_args()

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('- %s: %s', arg, getattr(args, arg))

    if args.nproc == 1:
        logger.warning('Running in single core mode. User nproc > 1 to run simulations concurrently')

    main(args.config_file, args.job_dir, args.nproc)

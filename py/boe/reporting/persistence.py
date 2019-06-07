import os
import os.path

import numpy as np

from boe.experimenting import record
from boe import logger


__AVG_REWARD__ = 'ar'
__TTL_REWARD__ = 'cr'
__PRB_SELECT__ = 'ps'
__UCB_CNFITV__ = 'ci'
__NMR_SIMULS__ = 'ns'

__FILE_NAME__ = 'exp-results'


def export(path, experiment_stats):
    export_path = os.path.join(path, __FILE_NAME__)
    logger.info('Exporting results to %s', export_path)

    data = {
        __AVG_REWARD__: experiment_stats.arms_average_rewards,
        __TTL_REWARD__: experiment_stats.arms_cumulative_rewards,
        __PRB_SELECT__: experiment_stats.arms_prob_selection,
        __UCB_CNFITV__: experiment_stats.arms_ucb,
        __NMR_SIMULS__: np.array([experiment_stats.num_sims], dtype=np.int32)
    }

    with open(export_path, 'wb') as fp:
        np.savez(fp, **data)


def load(path):
    export_path = os.path.join(path, __FILE_NAME__)
    logger.info('Loading results to %s', export_path)

    with open(export_path, 'rb') as fp:
        data = np.load(fp)

        return record.ExperimentStats(
            num_sims=data[__NMR_SIMULS__][0],
            arms_average_rewards=data[__AVG_REWARD__],
            arms_cumulative_rewards=data[__TTL_REWARD__],
            arms_prob_selection=data[__PRB_SELECT__],
            arms_ucb=data[__UCB_CNFITV__]
        )

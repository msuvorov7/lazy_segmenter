import argparse
import os
import sys
import time
import yaml
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.utils.logger import setup_logger
from src.train.train import train


def train_model(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = setup_logger('TRAIN', level=config['base']['log_level'])

    estimator_name = config['train']['estimator_name']
    logger.info(f'Estimator: {estimator_name}')

    with open(config['data_load']['X_np'], 'rb') as x_f:
        x = np.load(x_f)
    logger.info('Loaded train dataset')

    start_time = time.time()
    model = train(x, estimator_name)
    fitting_time = time.time() - start_time

    models_path = config['train']['model_path']
    joblib.dump(model, models_path)

    metrics = {
        estimator_name: fitting_time,
    }

    with open(config['reports']['metrics'], 'w') as m_f:
        json.dump(
            obj=metrics,
            fp=m_f,
            indent=4
        )
    logger.info('Saved metrics')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)

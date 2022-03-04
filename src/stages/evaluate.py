import argparse
import json

import yaml
import joblib
import numpy as np
from PIL import Image
from src.utils.logger import setup_logger

def evaluate_model(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = setup_logger('EVALUATE', level=config['base']['log_level'])

    model_path = config['train']['model_path']
    model = joblib.load(model_path)
    logger.info('Loaded model')

    with open(config['data_load']['X_np'], 'rb') as x_f:
        x = np.load(x_f)

    result = np.zeros(x.shape, dtype='uint8')
    colors = np.array(model.cluster_centers_, dtype='uint8')

    for ix in range(x.shape[0]):
        result[ix] = colors[model.labels_[ix]]

    with open(config['reports']['original_shape'], 'r') as mf:
        original_shape = json.load(mf)
    logger.info('Loaded test data')

    original_shape = (original_shape['height'], original_shape['width'], original_shape['channels'])
    result = np.reshape(result, original_shape)

    result = Image.fromarray(result)
    result.save(config['reports']['outputs'])
    logger.info('Saved result')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)

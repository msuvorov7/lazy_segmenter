import argparse
from PIL import Image
import yaml
import numpy as np
import json
from src.utils.logger import setup_logger


def data_load(config_path: str) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = setup_logger('DATA_LOAD', level=config['base']['log_level'])

    content = Image.open(config['data']['content'])
    X = np.array(content)

    original_shape = {
        'height': X.shape[0],
        'width': X.shape[1],
        'channels': X.shape[2],
    }
    with open(config['reports']['original_shape'], 'w') as mf:
        json.dump(
            obj=original_shape,
            fp=mf,
            indent=4
        )

    X = X.reshape(-1, 3)

    with open(config['data_load']['X_np'], 'wb') as x_f:
        np.save(x_f, X)

    logger.info('Saved raw data')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args = args_parser.parse_args()
    args_parser.add_argument('--config', dest='config', required=True)

    data_load(config_path=args.config)

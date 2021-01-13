import sys
sys.path.insert(0, '/home/ubuntu/experiments/mindsdb_native')
sys.path.insert(0, '/home/ubuntu/experiments/lightwood')

import logging
import pandas as pd
import numpy as np

from mindsdb import Predictor
from mindsdb_native.libs.controllers.functional import get_model_data

from frameworks.shared.callee import call_run, result
from amlb.results import save_predictions_to_file


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** MindsDB ****\n")

    target = dataset.target.name
    is_classification = config.type == 'classification'

    X_train = pd.DataFrame(dataset.train.X)
    y_train = pd.DataFrame(dataset.train.y)
    X_test = pd.DataFrame(dataset.test.X)
    y_test = pd.DataFrame(dataset.test.y)

    # fixed column names
    X_train.columns = 'c' + pd.Series([i for i in range(len(X_train.columns))]).astype(str)
    X_test.columns = X_train.columns

    # concat y to dataframe
    X_train[target] = y_train
    X_test[target] = y_test

    predictor = Predictor(name="MindsDB")
    predictor.quick_learn(from_data=X_train,
                          to_predict=target,
                          use_gpu=False,
                          advanced_args={
                              'use_mixers': 'NnMixer',
                              'output_class_distribution': True
                          },
                          stop_training_in_x_seconds=config.max_runtime_seconds)

    results = predictor.predict(when_data=X_test)._data
    truth = X_test[target].values

    if is_classification:
        model_data = get_model_data("MindsDB")
        beliefs = results[f'{target}_class_distribution']
        idx2cls = results[f'{target}_class_map']
        cls2idx = {v:k for k, v in idx2cls.items()}
        print(beliefs)
        classes = sorted(model_data['data_analysis_v2'][target]['histogram']['x'])
        print(classes)
        beliefs = np.array([[b[cls2idx[c]] for c in classes] for b in beliefs])
        preds = np.array([classes[int(float(i))] for i in results[target]])
        preds = pd.DataFrame(np.hstack([beliefs, preds.reshape(-1, 1)]))
        preds.columns = classes + ['predictions']
        preds['truth'] = truth
        print(preds)
        preds.to_csv(config.output_predictions_file, index=False)

    else:
        save_predictions_to_file(dataset=dataset,
                                 output_file=config.output_predictions_file,
                                 predictions=preds,
                                 truth=truth)

    return result()


if __name__ == '__main__':
    call_run(run)

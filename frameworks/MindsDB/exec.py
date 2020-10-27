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
                          stop_training_in_x_seconds=config.max_runtime_seconds)

    preds = predictor.quick_predict(when_data=X_test)
    preds = np.array(preds[target])
    truth = X_test[target].values

    if is_classification:
        model_data = get_model_data("MindsDB")
        classes = sorted(model_data['data_analysis_v2'][target]['histogram']['x'])
        preds_as_idxs = preds.astype(np.float).astype(np.int)
        one_hot_matrix = np.eye(len(classes))[preds_as_idxs]
        preds = pd.DataFrame(np.hstack([one_hot_matrix, preds.reshape(-1, 1)]))
        preds.columns = classes + ['predictions']
        preds['truth'] = truth
        preds.to_csv(config.output_predictions_file, index=False)

    else:
        save_predictions_to_file(dataset=dataset,
                                 output_file=config.output_predictions_file,
                                 predictions=preds,
                                 truth=truth)

    return result()


if __name__ == '__main__':
    call_run(run)

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
    predictor.learn(from_data=X_train, to_predict=target,
                    stop_training_in_x_seconds=config.max_runtime_seconds)

    predictions = predictor.predict(when_data=X_test)
    predictions = [x.explanation for x in predictions]

    preds = np.array([x[target]['predicted_value'] for x in predictions])
    truth = X_test[target].values
    # probs = np.array([x[target]['confidence'] for x in predictions]) # currently broken

    # custom csv when classifying
    if is_classification:
        model_data = get_model_data("MindsDB")
        classes = model_data['model_analysis'][0]['accuracy_histogram']['x']
        preds = preds.astype(np.float).astype(np.int)
        one_hot_matrix = np.eye(len(classes))[preds]
        preds = preds.astype(str)
        preds = pd.DataFrame(np.hstack([one_hot_matrix, preds.reshape(-1, 1)]))
        preds.columns = classes + ['predictions']
        preds['truth'] = truth.astype(np.float).astype(np.int).astype(str)
        preds.to_csv(config.output_predictions_file, index=False)

    # default for regression
    else:
        save_predictions_to_file(dataset=dataset,
                                 output_file=config.output_predictions_file,
                                 predictions=preds,
                                 # probabilities=probs,
                                 truth=truth)

    return result()


if __name__ == '__main__':
    call_run(run)

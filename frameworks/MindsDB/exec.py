import logging
import pandas as pd
import numpy as np

from mindsdb import Predictor
from scipy.io.arff import loadarff

from frameworks.shared.callee import call_run, result


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** MindsDB ****\n")

    target = dataset.target.name

    X_train = pd.DataFrame(loadarff(dataset.train.path)[0])
    X_test = pd.DataFrame(loadarff(dataset.test.path)[0])
    queries = X_test

    predictor = Predictor(name="MindsDB")
    predictor.learn(from_data=X_train, to_predict=target)

    predictions = predictor.predict(when_data=queries)
    predictions = [x.explanation for x  in predictions]

    preds = np.array([x[target]['predicted_value'] for x in predictions])
    probs = np.array([x[target]['confidence'] for x in predictions])
    truth = queries[target].values

    return result(output_file=config.output_predictions_file,
                  predictions=preds,
                  # probabilities=probs,
                  truth=truth
                  )


if __name__ == '__main__':
    call_run(run)

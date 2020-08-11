import logging
import pandas as pd

from mindsdb import Predictor
from scipy.io.arff import loadarff

from amlb.results import save_predictions_to_file
from frameworks.shared.callee import call_run, result  # , output_subdir, utils


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** MindsDB ****\n")

    target = dataset.target.name

    X_train = pd.DataFrame(loadarff(dataset.train.path)[0])
    X_test = pd.DataFrame(loadarff(dataset.test.path)[0])

    predictor = Predictor(name="MindsDB")
    predictor.learn(from_data=X_train, to_predict=target)

    save_predictions(predictor, X_test, target, config, dataset=dataset)
    # save_artifacts() here

    return result(output_file=config.output_predictions_file,
                  #predictions=None, probabilities=None, truth=None)
                  )

def save_predictions(predictor, queries, target, config, dataset, predictions_file=None, preview=True):
    predictions = predictor.predict(when_data=queries)
    predictions = [x.explanation for x  in predictions]

    pred_values = [x[target]['predicted_value'] for x in predictions]
    probabilities = [x[target]['confidence'] for x in predictions],
    truth = dataset[target]

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file if predictions_file is None else predictions_file,
                             probabilities=probabilities,
                             predictions=pred_values,
                             truth=truth,
                             preview=preview)


if __name__ == '__main__':
    call_run(run)

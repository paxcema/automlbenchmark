import logging
import pandas as pd
import numpy as np

from mindsdb import Predictor

from frameworks.shared.callee import call_run, result


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** MindsDB ****\n")

    target = dataset.target.name

    X_train = pd.DataFrame(dataset.train.X_enc)
    X_test = pd.DataFrame(dataset.test.X_enc)

    y_train = pd.DataFrame(dataset.train.y_enc)
    y_test = pd.DataFrame(dataset.test.y_enc)

    # fixed column names
    X_train.columns = 'c' + pd.Series([i for i in range(len(X_train.columns))]).astype(str)
    X_test.columns = X_train.columns

    # add y to dataframe
    X_train[target] = y_train
    X_test[target] = y_test

    predictor = Predictor(name="MindsDB")
    predictor.learn(from_data=X_train,
                    to_predict=target,
                    stop_training_in_x_seconds=config.max_runtime_seconds,
                    )

    predictions = predictor.predict(when_data=X_test)
    predictions = [x.explanation for x  in predictions]

    preds = np.array([x[target]['predicted_value'] for x in predictions])
    # probs = np.array([x[target]['confidence'] for x in predictions]) # these are broken currently
    truth = X_test[target].values

    return result(output_file=config.output_predictions_file,
                  predictions=preds,
                  # probabilities=probs,
                  truth=truth
                  )


if __name__ == '__main__':
    call_run(run)

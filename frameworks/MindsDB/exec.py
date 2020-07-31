import logging

import pandas as pd
from mindsdb import MindsDB, Predictor

from frameworks.shared.callee import call_run  # , result, output_subdir, utils


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** MindsDB ****\n")

    # get data
    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    df = pd.DataFrame([X_train, y_train])

    # join in a df

    # training_params = {k: v for k, v in config.framework_params.items()}
    mdb = MindsDB(name='benchmark')

    # train
    mdb.learn(from_data=X_train,
              to_predict=y_train)

    # test
    # result = 
    mdb.predict(when=dataset.test.y_enc)

    # return 


if __name__ == '__main__':
    call_run(run)
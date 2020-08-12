from amlb.utils import call_script_in_same_dir
from frameworks.shared.caller import run_in_venv
from amlb.datautils import impute


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset, config):
    X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
    data = dict(
            target=dict(name=dataset.target.name),
            train=dict(path=dataset.train.path,
                       X_enc=X_train_enc,
                       y_enc=dataset.train.y_enc
                       ),
            test=dict(path=dataset.test.path,
                      X_enc=X_test_enc,
                      y_enc=dataset.test.y_enc
                      )
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

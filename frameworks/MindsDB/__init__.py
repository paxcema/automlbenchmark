from amlb.utils import call_script_in_same_dir
from frameworks.shared.caller import run_in_venv


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset, config):
    data = dict(
            target=dict(name=dataset.target.name),
            train=dict(path=dataset.train.path),
            test=dict(path=dataset.test.path)
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

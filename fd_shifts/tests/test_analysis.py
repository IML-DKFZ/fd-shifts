# TODO: Implement unit tests as we refactor
import filecmp
import shutil
from pathlib import Path

from PIL import Image
import pytest
from omegaconf import OmegaConf

from fd_shifts import analysis
from fd_shifts.utils import exp_utils

DATA_DIR = Path(__file__).absolute().parent / "data"


@pytest.fixture(
    params=[
        "breeds_lr0.001_run0_do0",
        "svhn_openset_modelvit_bbvit_lr0.01_bs128_run0_do0_rew0",
        "cifar10_modeldg_bbvit_lr0.01_bs128_run0_do0_rew6",
        "cifar10_modelconfidnet_bbvit_lr0.01_bs128_run0_do1_rew2.2",
        "cifar10_modeldevries_bbvit_lr0.0003_bs128_run0_do0_rew2.2",
    ]
)
def data_dir(request):
    return DATA_DIR / request.param


@pytest.fixture
def tmp_test_dir(tmp_path, data_dir):
    # HACK: The analysis script deduces the exp name from the path
    tmp_test_path: Path = tmp_path / "tests" / "tests"
    tmp_test_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(data_dir / "config.yaml", tmp_test_path)
    shutil.copy(data_dir / "raw_output.npz", tmp_test_path)
    shutil.copy(data_dir / "external_confids.npz", tmp_test_path)

    if (data_dir / "raw_output_dist.npz").is_file():
        shutil.copy(data_dir / "raw_output_dist.npz", tmp_test_path)

    if (data_dir / "external_confids_dist.npz").is_file():
        shutil.copy(data_dir / "external_confids_dist.npz", tmp_test_path)

    return tmp_test_path


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_test_dir):
    monkeypatch.setenv("EXPERIMENT_ROOT_DIR", str(tmp_test_dir))
    monkeypatch.setenv("DATASET_ROOT_DIR", "")


def _check_dir_content(test_dir: Path, expected_dir: Path, snapshot):
    dcmp = filecmp.dircmp(test_dir, expected_dir, [".pytest_cache"])

    for file in dcmp.left_only:
        if ".npz" in file:
            # these were copied over as input
            continue

        if ".png" in file:
            # HACK: Matplotlib produces inconsistent pngs on different platforms
            assert Image.open(test_dir / file).size == snapshot
            assert file == snapshot

        elif ".csv" in str(file):
            assert (test_dir / file).read_text() == snapshot
        else:
            assert False


def test_analysis_blackbox(tmp_test_dir, data_dir, snapshot):
    exp_utils.set_seed(42)

    cf = OmegaConf.load(data_dir / "config.yaml")
    cf.exp.group_name = ""
    cf.test.dir = str(tmp_test_dir)

    analysis.main(
        in_path=cf.test.dir,
        out_path=cf.test.dir,
        query_studies=cf.eval.query_studies,
        add_val_tuning=cf.eval.val_tuning,
        threshold_plot_confid=None,
        cf=cf,
    )

    _check_dir_content(tmp_test_dir, data_dir, snapshot)

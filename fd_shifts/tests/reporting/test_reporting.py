# TODO: Implement unit tests as we refactor
import filecmp
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("agg")
from matplotlib.testing.decorators import image_comparison
import pytest

from fd_shifts import reporting

DATA_DIR = Path(__file__).absolute().parent / "data"


@pytest.fixture
def tmp_test_dir(tmp_path):
    for path in DATA_DIR.glob("*.csv"):
        shutil.copy(path, tmp_path)

    return tmp_path


def _check_dir_content(test_dir: Path, expected_dir: Path, snapshot):
    dcmp = filecmp.dircmp(test_dir, expected_dir, [".pytest_cache"])

    for file in dcmp.left_only:
        if ".png" in file:
            continue
        assert (test_dir / file).read_text() == snapshot(name=file)


@image_comparison(baseline_images=["main_plot", "vit_v_cnn"], extensions=["png"])
def test_reporting_blackbox(tmp_test_dir, snapshot):
    reporting.main(tmp_test_dir)
    _check_dir_content(tmp_test_dir, DATA_DIR, snapshot)

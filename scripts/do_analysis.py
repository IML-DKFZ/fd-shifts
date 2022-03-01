import argparse
import logging
from multiprocessing import Pool
from pathlib import Path
from random import shuffle
from datetime import datetime

from multiprocessing_logging import install_mp_handler
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from fd_shifts import analysis

# EXPERIMENT_ROOT_DIR=/home/t974t/Experiments/ DATASET_ROOT_DIR=/home/t974t/Data python -W ignore -m scripts.do_analysis -l debug -p /home/t974t/Experiments/vit/
# EXPERIMENT_ROOT_DIR=/home/t974t/Experiments/fd-shifts DATASET_ROOT_DIR=/home/t974t/Data python -W ignore -m scripts.do_analysis -l debug -p /home/t974t/Experiments/fd-shifts/

VIT_PATH = Path("~/Experiments/vit/").expanduser()
BASE_PATH = Path("~/Experiments/fd-shifts/").expanduser()


def run_analysis(path: Path):
    logger = logging.getLogger("fd_shifts")
    try:
        logger.info("Started analysis in %s", path)

        config_path = path.parent / "hydra" / "config.yaml"

        if not config_path.is_file():
            logger.error("File %s does not exist", config_path)
            return 1

        config = OmegaConf.load(config_path)

        analysis.main(
            in_path=config.test.dir,
            out_path=config.test.dir,
            query_studies=config.eval.query_studies,
            add_val_tuning=config.eval.val_tuning,
            threshold_plot_confid=None,
            cf=config,
        )

        logger.info("Finished analysis in %s", path)
        return 0
    except KeyboardInterrupt:
        logger.warning("keyboard interrupt")
    except:
        logger.exception("Exception occured in %s", path)
        logger.info("Abnormally finished analysis in %s", path)
    finally:
        return 1


def get_all_experiments(path: Path):
    return path.expanduser().glob("**/test_results")


def main():

    root_logger = logging.getLogger("fd_shifts")
    root_logger.setLevel(logging.INFO)

    console = Console()
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    root_logger.addHandler(console_handler)

    log_file = open(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_analysis.log", "w")
    file_handler = RichHandler(
        console=Console(file=log_file, force_terminal=True), rich_tracebacks=True
    )
    root_logger.addHandler(file_handler)

    install_mp_handler()

    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", action="store_true")
    parser.add_argument("-n", "--num-proc", type=int, default=None)
    parser.add_argument("-p", "--path", type=Path, default=VIT_PATH)
    parser.add_argument("-l", "--log-level", type=str, default="info")
    args = parser.parse_args()

    root_logger.setLevel(getattr(logging, args.log_level.upper()))

    tasks = list(get_all_experiments(args.path))
    shuffle(tasks)

    try:
        with Progress(console=console) as progress:
            task_id = progress.add_task("[cyan]Working...", total=len(tasks))
            with Pool(processes=args.num_proc) as pool:
                for _ in pool.imap_unordered(run_analysis, tasks):
                    progress.advance(task_id)
                pool.close()
                pool.join()
    except KeyboardInterrupt:
        root_logger.error("keyboard interrupt")
        return
    finally:
        log_file.close()


if __name__ == "__main__":
    main()

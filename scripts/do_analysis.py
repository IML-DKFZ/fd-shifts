import argparse
from multiprocessing import Pool, set_start_method
from pathlib import Path
from random import shuffle

import torch
from loguru import logger
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress
from threadpoolctl import threadpool_limits

from fd_shifts import analysis

# EXPERIMENT_ROOT_DIR=/home/t974t/Experiments/ DATASET_ROOT_DIR=/home/t974t/Data python -W ignore -m scripts.do_analysis -l debug -p /home/t974t/Experiments/vit/
# EXPERIMENT_ROOT_DIR=/home/t974t/Experiments/fd-shifts DATASET_ROOT_DIR=/home/t974t/Data python -W ignore -m scripts.do_analysis -l debug -p /home/t974t/Experiments/fd-shifts/

VIT_PATH = Path("~/Experiments/vit/").expanduser()
BASE_PATH = Path("~/Experiments/fd-shifts/").expanduser()


def set_logger(logger_):
    global logger
    logger = logger_


def run_analysis(path: Path):
    analysis.logger = logger
    analysis.eval_utils.logger = logger
    analysis.studies.logger = logger
    analysis.metrics.logger = logger
    try:
        logger.info("Started analysis in {}", path)

        config_path = path.parent / "hydra" / "config.yaml"

        if not config_path.is_file():
            logger.error("File %s does not exist", config_path)
            return 1

        config = OmegaConf.load(config_path)
        config.exp.name = str(path.parts[-2])

        if "fd-shifts_64" in str(path):
            if hasattr(config.eval.query_studies, "noise_study"):
                config.eval.query_studies.pop("noise_study")

        analysis.main(
            in_path=config.test.dir,
            out_path=config.test.dir,
            query_studies=config.eval.query_studies,
            add_val_tuning=config.eval.val_tuning,
            threshold_plot_confid=None,
            cf=config,
        )

        logger.info("Finished analysis in {}", path)
        logger.complete()
        return 0
    except KeyboardInterrupt:
        logger.warning("keyboard interrupt")
    except:
        logger.exception("Exception occured in {}", path)
        logger.info("Abnormally finished analysis in {}", path)
    finally:
        logger.complete()
        return 1


def get_all_experiments(path: Path):
    # return filter(lambda x: "svhn" not in str(x), path.expanduser().glob("**/test_results"))
    return path.expanduser().glob("**/test_results")
    # return path.expanduser().glob("svhn_openset*/test_results")


def main():
    set_start_method("spawn")
    torch.set_num_threads(1)

    console = Console(stderr=True)
    logger.remove()  # Remove default 'stderr' handler

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    logger.add(
        lambda m: console.print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="INFO",
    )
    logger.add("{time}_do-analysis.log", enqueue=True, level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", action="store_true")
    parser.add_argument("-n", "--num-proc", type=int, default=None)
    parser.add_argument("-p", "--path", type=Path, default=VIT_PATH)
    parser.add_argument("-l", "--log-level", type=str, default="info")
    args = parser.parse_args()

    try:
        with Progress(console=console) as progress:
            task_id = progress.add_task(
                "[cyan]Working...", total=len(list(get_all_experiments(args.path)))
            )
            with threadpool_limits(limits=1, user_api="blas"):
                with Pool(
                    processes=args.num_proc,
                    initializer=set_logger,
                    initargs=(logger,),
                    maxtasksperchild=1,
                ) as pool:
                    for _ in pool.imap_unordered(
                        run_analysis, get_all_experiments(args.path),
                    ):
                        progress.advance(task_id)
    except KeyboardInterrupt:
        logger.error("keyboard interrupt")
        return


if __name__ == "__main__":
    main()

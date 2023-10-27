from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

# CPU device metrics from lightning 2.1.0, as 1.6.5 doesn't have them yet
_CPU_VM_PERCENT = "cpu_vm_percent"
_CPU_PERCENT = "cpu_percent"
_CPU_SWAP_PERCENT = "cpu_swap_percent"


def get_cpu_stats() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        raise ModuleNotFoundError(
            "Fetching CPU device stats requires `psutil` to be installed."
        )

    return {
        _CPU_VM_PERCENT: psutil.virtual_memory().percent,
        _CPU_PERCENT: psutil.cpu_percent(),
        _CPU_SWAP_PERCENT: psutil.swap_memory().percent,
    }


# Adapted from lightning 2.1.0, as it does not have nvidia-smi based stats
class DeviceStatsMonitor(Callback):
    r"""Automatically monitors and logs device stats during training, validation and testing stage.
    ``DeviceStatsMonitor`` is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Args:
        cpu_stats: if ``None``, it will log CPU stats only if the accelerator is CPU.
            If ``True``, it will log CPU stats regardless of the accelerator.
            If ``False``, it will not log CPU stats regardless of the accelerator.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.
        ModuleNotFoundError:
            If ``psutil`` is not installed and CPU stats are monitored.

    Example::

        from lightning import Trainer
        from lightning.pytorch.callbacks import DeviceStatsMonitor
        device_stats = DeviceStatsMonitor()
        trainer = Trainer(callbacks=[device_stats])

    """

    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        self._cpu_stats = cpu_stats

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        if stage != "fit":
            return

        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `DeviceStatsMonitor` callback with `Trainer(logger=False)`."
            )

        # warn in setup to warn once
        device = trainer.strategy.root_device

    def _get_and_log_device_stats(self, trainer: "pl.Trainer", key: str) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        device = trainer.strategy.root_device
        if self._cpu_stats is False and device.type == "cpu":
            # cpu stats are disabled
            return

        # device_stats = trainer.accelerator.get_device_stats(device)
        device_stats = {}

        if self._cpu_stats:
            # Don't query CPU stats twice if CPU is accelerator
            device_stats.update(get_cpu_stats())

        if device.type == "cuda":
            from pytorch_lightning.accelerators.gpu import get_nvidia_gpu_stats

            gpu_stats = get_nvidia_gpu_stats(device)
            if any(map(lambda k: k in device_stats, gpu_stats.keys())):
                raise RuntimeError("Replacing Stats")

            device_stats.update(gpu_stats)

        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(
                device_stats, f"{self.__class__.__qualname__}.{key}", separator
            )
            logger.log_metrics(
                prefixed_device_stats,
                # step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_start")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_end")

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_start")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_end")

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_start")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "batch_end")


def _prefix_metric_keys(
    metrics_dict: dict[str, float], prefix: str, separator: str
) -> dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}

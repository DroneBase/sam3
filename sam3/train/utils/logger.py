# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from sam3.train.utils.train_utils import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class WandBLogger:
    """
    A logger that writes scalars to Weights & Biases (WandB).
    Only logs on rank 0.

    Metrics are buffered per training step and flushed together so that
    WandB receives one ``wandb.log()`` call per step.  This avoids
    non-monotonic step warnings and ensures all metrics for a given step
    appear together — which is what makes real-time charts work correctly
    on the WandB dashboard.

    The trainer calls ``log(name, data, step)`` many times at the same
    training step (once per loss component, optimizer param, etc.) and
    ``log_dict(payload, step)`` once per epoch for aggregated metrics.
    Without buffering each individual ``wandb.log()`` call would either
    overwrite earlier data at the same step or trigger step-order warnings.

    Epoch-level metrics from ``log_dict`` are logged with a prefixed key
    ``epoch`` so they can be plotted against the epoch axis on WandB using
    custom x-axis settings.
    """

    def __init__(self, project: str = None, entity: str = None, name: str = None,
                 config: Dict = None, tags: list = None, notes: str = None, **kwargs):
        self._run = None
        self._buffer: Dict[str, Any] = {}
        self._buffer_step: Optional[int] = None
        _, self._rank = get_machine_local_and_dist_rank()
        if self._rank == 0:
            try:
                import wandb
                import os
                api_key = os.environ.get("WANDB_API_KEY", "")
                if not api_key:
                    logging.warning("WANDB_API_KEY not set — WandB logging disabled.")
                    return

                init_kwargs = {}
                if project:
                    init_kwargs["project"] = project
                if entity:
                    init_kwargs["entity"] = entity
                if name:
                    init_kwargs["name"] = name
                if config:
                    init_kwargs["config"] = config
                if tags:
                    init_kwargs["tags"] = tags
                if notes:
                    init_kwargs["notes"] = notes

                # If a run is already active (e.g. from the wrapper script), reuse it
                if wandb.run is not None:
                    self._run = wandb.run
                    logging.info(f"WandB: reusing existing run {self._run.url}")
                else:
                    self._run = wandb.init(**init_kwargs)
                    logging.info(f"WandB: initialized new run {self._run.url}")

                if self._run is not None:
                    # Let WandB know that "epoch" is a custom x-axis so
                    # epoch-level charts auto-configure correctly.
                    wandb.define_metric("epoch")
                    wandb.define_metric("epoch/*", step_metric="epoch")
            except Exception as e:
                logging.warning(f"Failed to initialize WandB logger: {e}")
                self._run = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Send the buffered metrics to WandB in a single call."""
        if not self._run or not self._buffer:
            return
        import wandb
        wandb.log(self._buffer, step=self._buffer_step)
        self._buffer = {}
        self._buffer_step = None

    # ------------------------------------------------------------------
    # Public API (matches TensorBoardLogger interface)
    # ------------------------------------------------------------------
    def log(self, name: str, data: Scalar, step: int) -> None:
        """Buffer a single scalar.  Metrics are flushed when the step changes."""
        if not self._run:
            return
        # If the step changed, flush the previous batch first
        if self._buffer_step is not None and step != self._buffer_step:
            self._flush()
        self._buffer_step = step
        self._buffer[name] = data

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Log a batch of epoch-level metrics immediately.

        Epoch-level metrics use a separate ``epoch`` x-axis so they
        don't collide with per-iteration training-step metrics.
        """
        if not self._run:
            return
        # Flush any pending per-step metrics first
        self._flush()
        import wandb
        epoch_payload = {"epoch": step}
        for k, v in payload.items():
            epoch_payload[f"epoch/{k}"] = v
        wandb.log(epoch_payload)

    def log_hparams(self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]) -> None:
        if not self._run:
            return
        import wandb
        wandb.config.update(hparams, allow_val_change=True)

    def close(self) -> None:
        self._flush()
        if self._run:
            import wandb
            wandb.finish()
            self._run = None


def make_wandb_logger(project: str = None, entity: str = None, name: str = None,
                      config: Dict = None, tags: list = None, notes: str = None,
                      **kwargs) -> WandBLogger:
    """Factory function for creating a WandBLogger (mirrors make_tensorboard_logger)."""
    return WandBLogger(
        project=project, entity=entity, name=name,
        config=config, tags=tags, notes=notes, **kwargs
    )


class Logger:
    """
    A logger class that can interface with multiple loggers.
    Supports TensorBoard and Weights & Biases (WandB).
    """

    def __init__(self, logging_conf):
        # allow turning off TensorBoard with "should_log: false" in config
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

        # WandB logger (optional)
        wb_config = logging_conf.wandb_writer
        self.wb_logger = instantiate(wb_config) if wb_config else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wb_logger:
            self.wb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if self.wb_logger:
            self.wb_logger.log_hparams(hparams, meters)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()

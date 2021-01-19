# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import logging
import os
import platform
import threading
import tracemalloc
from distutils import spawn
from subprocess import PIPE, CalledProcessError, Popen

logger = logging.getLogger(__name__)


class MemoryTrace(threading.Thread):
    # https://tech.buzzfeed.com/finding-and-fixing-memory-leaks-in-python-413ce4266e7d

    def __init__(self, enabled, delay=300, top=10, trace=1):
        """
        Log memory usage on a delay.

        :param delay: in seconds (int)
        :param top: number of top allocations to list (int)
        :param trace: number of top allocations to trace (int)
        :return: None
        """
        super().__init__()

        self.enabled = enabled
        self.delay = delay
        self.top = top
        self.trace = trace

        self.setDaemon(True)
        self._event = threading.Event()

        self._running = False
        self._start_stats = None
        self._prev_stats = None
        self._snapshot_lock = threading.Lock()

        self.start()

    def run(self):
        if self.enabled:
            self._running = True

            logger.debug("Starting MemoryTrace")
            tracemalloc.start(25)
            self._start_stats = tracemalloc.take_snapshot()
            self._prev_stats = self._start_stats

            while self._running:
                self._event.wait(self.delay)
                self.snapshot()

    def snapshot(self, title="Snapshot"):
        if self.enabled:
            with self._snapshot_lock:
                current = tracemalloc.take_snapshot()

                if title:
                    logger.debug("------ %s ------", title)

                # compare current snapshot to starting snapshot
                stats = current.compare_to(self._start_stats, "filename")

                # compare current snapshot to previous snapshot
                prev_stats = current.compare_to(self._prev_stats, "lineno")

                logger.debug("GPU Stats")
                for gpu in get_gpu_stats():
                    logger.debug(
                        "gpu_stats id={}, name={}, mem_used={}, mem_total={}, mem_util={} %, volatile_gpu={}, temp={} C".format(
                            gpu.id,
                            gpu.name,
                            gpu.memory_used,
                            gpu.memory_total,
                            int(gpu.memory_util * 100),
                            gpu.util,
                            gpu.temperature,
                        )
                    )

                logger.debug("Top Diffs since Start")
                for i, stat in enumerate(stats[: self.top], 1):
                    logger.debug("top_diffs i=%d, stat=%s", i, str(stat))

                logger.debug("Top Incremental")
                for i, stat in enumerate(prev_stats[: self.top], 1):
                    logger.debug("top_incremental i=%d, stat=%s", i, str(stat))

                logger.debug("Top Current")
                for i, stat in enumerate(current.statistics("filename")[: self.top], 1):
                    logger.debug("top_current i=%d, stat=%s", i, str(stat))

                traces = current.statistics("traceback")
                for stat in traces[: self.trace]:
                    logger.debug(
                        "traceback memory_blocks=%d, size_kB=%d",
                        stat.count,
                        stat.size / 1024,
                    )
                    for line in stat.traceback.format():
                        logger.debug(line)

                self._prev_stats = current

    def stop(self):
        logger.debug("Stopping MemoryTrace")
        self._running = False
        self._event.set()


class GPU:
    """
    Storage class for holding onto GPU status.

    Attributes
    ---
    id: int
        GPU device id.
    uuid: str
        GPU device uuid.
    util: str
        Volatile GPU Utilization
    memory_util: float
        Percentage of total device memory being utilized.
    memory_total: str
        Total device memory with units.
    memory_used: str
        Amount of device memory being used with units.
    memory_free: str
        Amount of free device memory with units.
    driver: str
        Device driver version.
    name: str
        Device name
    serial: str
        Device serial number
    display_mode: str
        Whether display mode is Enabled or Disabled
    display_acitve: str
        Whether display active is Enabled or Disabled
    temperature: int
        GPU temperature in Celsius.
    """

    def __init__(
        self,
        id,
        uuid,
        util,
        memory_total,
        memory_used,
        memory_free,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
        temp_gpu,
    ):
        self.id = id
        self.uuid = uuid
        self.util = util
        self.memory_util = float(memory_used.split(" ")[0]) / float(
            memory_total.split(" ")[0]
        )
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.memory_free = memory_free
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu


def get_gpu_stats():
    """
    Calls `nvidia-smi` cli to query gpu status and store it in an object.

    Attributes
    ---
    id: int
        GPU device id.
    uuid: str
        GPU device uuid.
    util: str
        Volatile GPU Utilization
    memory_util: float
        Percentage of total device memory being utilized.
    memory_total: str
        Total device memory with units.
    memory_used: str
        Amount of device memory being used with units.
    memory_free: str
        Amount of free device memory with units.
    driver: str
        Device driver version.
    name: str
        Device name
    serial: str
        Device serial number
    display_mode: str
        Whether display mode is Enabled or Disabled
    display_acitve: str
        Whether display active is Enabled or Disabled
    temperature: int
        GPU temperature in Celsius.

    :return: List of GPU objects.
    """
    if platform.system() == "Windows":
        nvidia_smi = spawn.find_executable("nvidia-smi")
        if not nvidia_smi:
            nvidia_smi = (
                "{}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe".format(
                    os.environ["systemdrive"]
                )
            )
    else:
        nvidia_smi = "nvidia-smi"

    try:
        stdout, _ = Popen(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                "--format=csv,noheader",
            ],
            stdout=PIPE,
        ).communicate()
    except (CalledProcessError, FileNotFoundError):
        logger.warning(
            "Couldn't locate CUDA devices using nvidia-smi. Check that nvidia-smi is installed correctly."
        )
        return []

    devices = stdout.decode("UTF-8").split(os.linesep)
    gpus = []

    for i in range(len(devices) - 1):
        vals = devices[i].split(", ")

        id = vals[0]
        uuid = vals[1]
        gpu_util = vals[2]
        mem_total = vals[3]
        mem_used = vals[4]
        mem_free = vals[5]
        driver = vals[6]
        gpu_name = vals[7]
        serial = vals[8]
        display_active = vals[9]
        display_mode = vals[10]
        temp_gpu = vals[11]

        gpus.append(
            GPU(
                id,
                uuid,
                gpu_util,
                mem_total,
                mem_used,
                mem_free,
                driver,
                gpu_name,
                serial,
                display_mode,
                display_active,
                temp_gpu,
            )
        )

    return gpus

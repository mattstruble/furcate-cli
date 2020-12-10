# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import os
import platform
from distutils import spawn
from subprocess import PIPE, CalledProcessError, Popen


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
    except CalledProcessError:
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

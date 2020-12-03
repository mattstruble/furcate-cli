# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import os
import platform
from distutils import spawn
from subprocess import Popen, PIPE

class GPU:
    def __init__(self, id, uuid, util, memory_total, memory_used, memory_free, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        self.id = id
        self.uuid = uuid
        self.util = util.replace(' ', '')
        self.memory_util = float(memory_used.split(' ')[0])/float(memory_total.split(' ')[0])
        self.memory_total = memory_total.replace(' ', '')
        self.memory_used = memory_used.replace(' ', '')
        self.memory_free = memory_free.replace(' ', '')
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu

def get_gpus(framework):
    if framework == 'tf':
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
    else:
        raise TypeError("The supplied framework '{}' is not supported.".format(framework))

    return gpus

def set_gpus(index, framework):
    gpus = get_gpus(framework)

    if framework == 'tf':
        import tensorflow as tf
        tf.config.set_visible_devices(gpus[int(index)], 'GPU')
    else:
        raise TypeError("The supplied framework '{}' is not supported.".format(framework))

def get_gpu_stats():
    if platform.system() == "Windows":
        nvidia_smi = spawn.find_executable("nvidia-smi")
        if not nvidia_smi:
            nvidia_smi = "{}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe".format(os.environ['systemdrive'])
    else:
        nvidia_smi = "nvidia-smi"

    try:
        p = Popen([nvidia_smi,"--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []

    devices = stdout.decode('UTF-8').split(os.linesep)
    gpus = []

    for i in range(len(devices)-1):
        vals = devices[i].split(', ')

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

        gpus.append(GPU(id, uuid, gpu_util, mem_total, mem_used, mem_free, driver, gpu_name, serial, display_mode,
                        display_active, temp_gpu))

    return gpus
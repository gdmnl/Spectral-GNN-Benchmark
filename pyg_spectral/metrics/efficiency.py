import time
import resource
import torch
from torch.nn import Module


class Stopwatch(object):
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def pause(self) -> float:
        """Pause clocking and return elapsed time"""
        self.elapsed_sec += time.time() - self.start_time
        self.start_time = None
        return self.elapsed_sec

    def lap(self) -> float:
        """No pausing, return elapsed time"""
        return time.time() - self.start_time + self.elapsed_sec

    def reset(self):
        self.start_time = None
        self.elapsed_sec = 0

    @property
    def time(self) -> float:
        return self.elapsed_sec


class Accumulator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, val: float, count: int=1):
        self.val += val
        self.count += count
        return self.val

    @property
    def avg(self) -> float:
        return self.val / self.count


def memory_ram() -> float:
    r"""Current RAM usage in GB
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def memory_cuda(dev) -> float:
    r"""Current CUDA memory usage in GB
    """
    return torch.cuda.max_memory_allocated(dev) / 2**30


def get_num_params(model: Module) -> float:
    r"""Number of module parameters
    """
    num_paramst = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    num_params = sum([param.nelement() for param in model.parameters()])
    num_bufs = sum([buf.nelement() for buf in model.buffers()])
    # return num_paramst/1e6, num_params/1e6, num_bufs/1e6
    return num_paramst/1e6


def get_mem_params(model: Module) -> float:
    mem_paramst = sum([param.nelement()*param.element_size() for param in model.parameters() if param.requires_grad])
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    # return mem_paramst/(1024**2), mem_params/(1024**2), mem_bufs/(1024**2)
    return (mem_params+mem_bufs)/(1024**2)

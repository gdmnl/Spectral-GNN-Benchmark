# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-10-08
File: efficiency.py
"""
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
        self.data = 0
        self.count = 0

    def update(self, val: float, count: int=1):
        self.data += val
        self.count += count
        return self.data

    @property
    def avg(self) -> float:
        return self.data / self.count


class NumFmt(object):
    def __init__(self, data: float = None, base: int = 10, suffix: str = ''):
        assert base in [2, 10]
        self.base = base
        self.base_exp = 3 if base == 10 else 10
        self.units = ['', 'K', 'M', 'G', 'T']
        self.suffix = suffix
        self.set(data)

    def set(self, num: float):
        if num is not None:
            self.data = num

    def update(self):
        pass

    def get_unit(self, num) -> str:
        if num == 0:
            return 0, self.units[self.base][0]
        num = abs(num)

        if self.base == 2:
            exp = int((num.bit_length() - 1) / 10)
        else:
            exp = int(len(str(num)) / 3)
        return self.units[exp]

    def get(self, unit: str = None) -> float:
        if unit is None:
            unit = self.get_unit(self.data)
        assert unit in self.units, f'Unit should be one of {self.units}'
        exp = self.units.index(unit)
        return self.data / (self.base ** (exp * self.base_exp))

    def __str__(self) -> str:
        unit = self.get_unit(self.data)
        return f'{self.get(unit):.2f} {unit}{self.suffix}'


class MemoryRAM(NumFmt):
    r"""Memory usage of current process in RAM.
    """
    def __init__(self):
        super(MemoryRAM, self).__init__(base=2, suffix='B')
        self.update()

    def update(self):
        self.set(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 2**10)


class MemoryCUDA(NumFmt):
    r"""Memory usage of current process in CUDA.
    """
    def __init__(self):
        super(MemoryCUDA, self).__init__(base=2, suffix='B')
        self.update()

    def update(self):
        self.set(torch.cuda.max_memory_allocated())


class ParamNumel(NumFmt):
    r"""Number of learnable parameters in an nn.Module.
    """
    def __init__(self, module: Module = None):
        super(ParamNumel, self).__init__(base=10, suffix='')
        if module is not None:
            self.update(module)

    def update(self, module: Module):
        num_paramst = sum([param.nelement() for param in module.parameters() if param.requires_grad])
        # num_params = sum([param.nelement() for param in module.parameters()])
        # num_bufs = sum([buf.nelement() for buf in module.buffers()])
        self.set(num_paramst)


class ParamMemory(NumFmt):
    r"""Memory usage of parameters in an nn.Module.
    """
    def __init__(self, module: Module = None):
        super(ParamMemory, self).__init__(base=2, suffix='B')
        if module is not None:
            self.update(module)

    def update(self, module: Module):
        # mem_paramst = sum([param.nelement()*param.element_size() for param in module.parameters() if param.requires_grad])
        mem_params = sum([param.nelement()*param.element_size() for param in module.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in module.buffers()])
        self.set(mem_params + mem_bufs)

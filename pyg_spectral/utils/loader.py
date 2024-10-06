import importlib
from functools import wraps


def load_import(class_name, module_name):
    r"""Simple dynamic import for 'module.class'"""
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    if isinstance(class_obj, type):
        return class_obj


def resolve_func(nargs=1):
    """Enable calling the return function with additional arguments.

    Args:
        nargs: The number of arguments to pass to the decorated function. Arguments
        beyond this number will be passed to the return function.
    Examples:
        ```python
        @resolve_func(1)
        def foo(bar):
            return bar
        foo(1)  # 1
        foo(lambda x: x+1, 2)  # 3
        ```
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*inputs):
            if len(inputs) <= nargs:
                return func(*inputs[:nargs])
            else:
                ret = func(*inputs[:nargs])
                if callable(ret):
                    return ret(*inputs[nargs:])
                return ret
        return wrapper
    return decorator


class CallableDict(dict):
    def __call__(self, key, *args):
        def _get_callable(key):
            ret = self.get(key, None)
            if callable(ret):
                return ret(*args)
            return ret

        if not key in self:
            if ',' in key:
                return [_get_callable(k) for k in key.split(',')]
            raise ValueError(f"Key '{key}' not found in {self.keys()}.")
        return _get_callable(key)

    @classmethod
    def to_callableVal(cls, dct, keys=None):
        keys = keys or dct.keys()
        for key in keys:
            if isinstance(dct[key], dict):
                dct[key] = cls(dct[key])
        return dct

    @classmethod
    def to_subcallableVal(cls, dct, keys=[]):
        for key in dct:
            if key in keys:
                dct[key] = cls.to_callableVal(dct[key])
            else:
                if isinstance(dct[key], dict):
                    dct[key] = cls(dct[key])
        return dct

import importlib
from functools import wraps


def load_import(class_name, module_name):
    r"""Simple dynamic import for ``module.class``"""
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    if isinstance(class_obj, type):
        return class_obj


def resolve_func(nargs=1):
    """Enable calling the return function with additional arguments.

    Args:
        nargs: The number of arguments to pass to the decorated function. Arguments
        beyond this number will be passed to the return function.
    Examples::

        @resolve_func(1)
        def foo(bar):
            return bar

        >>> foo(1)
        1
        >>> foo(lambda x: x+1, 2)
        3
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
    """
    A dictionary subclass that allows its values to be called as functions.
    """

    def __call__(self, key, *args):
        r"""Get key value and call it with args if it is callable.

        Args:
            key: The key to get the value from.
            *args: Arguments to pass to the indexed value if it is callable.
        Returns:
            ret (Any | list): If the value is callable, returns the result of
            calling it with args. Otherwise, returns the value.
        """
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
    def to_callableVal(cls, dct, keys:list=None, reckeys:list=[]):
        r"""Converts the sub-dictionaries of the specified keys in the
        dictionary to :class:`CallableDict`.

        Args:
            dct (dict): The dictionary to convert.
            keys (list[str]): The keys to convert. If None, converts all sub-dictionaries.
            reckeys (list[str]): The keys to recursively convert sub-sub-dictionaries.
        Returns:
            dct (dict): The dictionary with the specified keys converted to :class:`CallableDict`.
        Examples::

            dct = {'key0': Dict0, 'key1': Dict1, 'key2': Dict2}
            dct = CallableDict.to_callableVal(dct, keys=['key1'], reckeys=['key2'])
            # dct = {'key0': Dict0, 'key1': CallableDict1, 'key2': Dict2},
            # and each sub-dictionary in 'key2' is converted to CallableDict.
        """
        keys = keys or dct.keys()
        for key in dct:
            if key in reckeys:
                dct[key] = cls.to_callableVal(dct[key])
            elif key in keys:
                if isinstance(dct[key], dict):
                    dct[key] = cls(dct[key])
        return dct

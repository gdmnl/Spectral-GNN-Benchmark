import importlib


def load_import(class_name, module_name):
    r"""Simple dynamic import for 'module.class'"""
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    if isinstance(class_obj, type):
        return class_obj

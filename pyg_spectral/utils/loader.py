
def load_import(class_name, module_name):
    r"""Simple hack for dynamic importing module.class"""
    module = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(module, class_name)
    if isinstance(class_obj, type):
        return class_obj

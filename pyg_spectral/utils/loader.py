
def load_import(class_name, module_name):
    # Simple hack for dynamic import
    module = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(module, class_name)
    if isinstance(class_obj, type):
        return class_obj

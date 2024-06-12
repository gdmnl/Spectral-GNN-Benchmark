try:
    from .prop_cppext import PyPropComp
    WITH_CPP_PROPAGATIONS = True
except ImportError:
    PyPropComp = object
    WITH_CPP_PROPAGATIONS = False

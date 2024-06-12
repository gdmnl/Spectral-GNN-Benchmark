import os
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
    import eigency
    WITH_CPP = True

    ext_modules = cythonize(Extension(
        name='pyg_spectral.propagations.prop_cppext',
        sources=['pyg_spectral/propagations/prop_cppext.pyx'],
        language='c++',
        extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
        include_dirs=[".", "module-dir-name"] + eigency.get_includes(),
        optional=True,
    ))
except ImportError:
    WITH_CPP = False
    ext_modules = []

FLAG_CPP = os.getenv('PSFLAG_CPP', '0') == '1'

setup(
    packages=find_packages(),
    ext_modules=ext_modules if WITH_CPP and FLAG_CPP else [],
)
# TODO: [optional benckmark](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies)

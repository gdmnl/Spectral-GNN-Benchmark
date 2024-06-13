import os
from setuptools import setup, find_packages, Extension

setup_requires = ['setuptools', 'wheel']

FLAG_CPP = os.getenv('PSFLAG_CPP', '0') == '1'
if FLAG_CPP:
    try:
        from Cython.Build import cythonize
        import eigency
        WITH_CPP = True

        setup_requires += ['cython', 'eigency']
        ext_modules = cythonize(Extension(
            name='pyg_spectral.propagations.prop_cppext',
            sources=['pyg_spectral/propagations/prop_cppext.pyx'],
            language='c++',
            extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
            include_dirs=[".", "module-dir-name"] + eigency.get_includes(),
            optional=True,
        ))
    except ImportError:
        import warnings
        warnings.warn("Cython or Eigen is not installed, continue without cpp.", category=ImportWarning, stacklevel=2)
        WITH_CPP = False
        ext_modules = []
else:
    ext_modules = []

setup(
    setup_requires=setup_requires,
    packages=find_packages(),
    ext_modules=ext_modules,
)
# FEATURE: [optional benckmark](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies)

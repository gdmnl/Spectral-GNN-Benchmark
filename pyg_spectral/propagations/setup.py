from setuptools import setup, Extension
from Cython.Build import cythonize
import eigency
import os

home = os.path.expanduser("~")
path_eigen = os.path.join(home, ".local/include/eigen-3.4.0")

setup(
    author='nyLiao',
    version='0.0.1',
    install_requires=['Cython>=0.2.15', 'eigency>=1.77'],
    python_requires='>=3',
    ext_modules=cythonize(Extension(
        name='prop_cppext',
        sources=['prop_cppext.pyx'],
        language='c++',
        extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
        include_dirs=[".", "module-dir-name"] + eigency.get_includes()[:2] + [path_eigen],
    ))
)

from setuptools import setup, Extension
import numpy

module = Extension(
    "espprc",
    sources=["espprc_module.c"],
    include_dirs=[numpy.get_include()]
)

setup(
    name="espprc",
    version="1.0",
    ext_modules=[module]
)

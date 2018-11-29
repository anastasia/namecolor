import os
from distutils.core import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname + '.md')).read()


setup(
    name="namecolor",
    version="0.0.1",
    author="Anastasia Aizman",
    author_email="anastasia.aizman@gmail.com",
    description="RNN — coloring words",
    keywords="namecolor",
    py_modules=["namecolor"],
    url="",
    long_description=read("README"),
)

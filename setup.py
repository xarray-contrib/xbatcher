 #!/usr/bin/env python
import os
import re
import sys
import warnings

from setuptools import setup, find_packages

VERSION = '0.1.0'
DISTNAME = 'xbatcher'
LICENSE = 'Apache'
AUTHOR = 'xbatcher Developers'
AUTHOR_EMAIL = 'rpa@ldeo.columbia.edu'
URL = 'https://github.com/xgcm/xbatcher'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xarray', 'dask', 'numpy']
SETUP_REQUIRES = []
TESTS_REQUIRE = ['pytest >= 2.8', 'coverage']

DESCRIPTION = "Batch generation from xarray dataset"
def readme():
    return "TODO"
    #with open('README.rst') as f:
    #    return f.read()


setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=readme(),
      install_requires=INSTALL_REQUIRES,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages())

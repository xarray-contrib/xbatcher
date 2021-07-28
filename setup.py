#!/usr/bin/env python
import os

from setuptools import find_packages, setup

VERSION = '0.1.0'
DISTNAME = 'xbatcher'
LICENSE = 'Apache'
AUTHOR = 'xbatcher Developers'
AUTHOR_EMAIL = 'rpa@ldeo.columbia.edu'
URL = 'https://github.com/pangeo-data/xbatcher'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

DESCRIPTION = 'Batch generation from xarray dataset'

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open('dev-requirements.txt') as f:
    test_requires = f.read().strip().split('\n')

if os.path.exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''


setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=long_description,
    python_requires='>=3.7',
    install_requires=install_requires,
    url=URL,
    packages=find_packages(),
)

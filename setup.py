

from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import importlib
import logging
import shutil

def _safe_read_lines(f):        
    with open(f) as in_f:
        r = in_f.readlines()
    r = [l.strip() for l in r]
    return r

console_scripts = [
    'create-extended-listfile=mimic_preprocessing.create_extended_listfile:main',
    'create-extended-mimic-dataset=mimic_preprocessing.create_extended_mimic_dataset:main',
    'create-mimic-notes-bow=mimic_preprocessing.create_mimic_notes_bow:main',
    'extract-mimic-time-series-features=mimic_preprocessing.extract_mimic_time_series_features:main',
]

install_requires = _safe_read_lines("./requirements.txt")

tests_require = []
extras = {}

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Programming Language :: Python :: 3 :: Only',
]

def _post_install(self):
    import site
    importlib.reload(site)


class my_install(_install):
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _install.run(self)
        _post_install(self)

class my_develop(_develop):  
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _develop.run(self)
        _post_install(self)

def readme():
    with open('README.md') as f:
        return f.read()

def description():
    description = ("This package includes helper scripts to add additional "
        "data modalities to the MIMIC-III dataset.")
    return description

setup(
    name='mimic_preprocessing',
    version='0.1.0',
    description=description(),
    long_description=readme(),
    keywords="electronic health records mimic-iii preprocessing",
    url="https://github.com/bmmalone/mimic-preprocessing",
    author="NLE",
    author_email="brandon.malone@neclab.eu",
    license='NLE ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY',
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={'install': my_install,  # override install
                'develop': my_develop   # develop is used for pip install -e .
    },
    include_package_data=True,
    tests_require=tests_require,
    extras_require=extras,
    entry_points = {
        'console_scripts': console_scripts
    },
    zip_safe=False,
    classifiers=classifiers,    
)

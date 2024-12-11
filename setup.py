from setuptools import find_packages, setup

setup(
    name='wrf_io',
    packages=find_packages(include=['wrf_io']),
    version='0.1.0',
    description='Pre and post-processing for WRF',
    author='Storm Mata',
    install_requires=[],
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest==4.4.1'],
    # test_suite='tests',
)
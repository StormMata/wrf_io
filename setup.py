from setuptools import find_packages, setup

setup(
    name='wrf_io',
    packages=find_packages(include=['wrf_io']),
    version='2.0.0',
    description='Pre and post-processing for WRF',
    author='Storm Mata',
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "netCDF4",  
        "rich",
        "Pillow",
        "glob2",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
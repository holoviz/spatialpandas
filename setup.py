from setuptools import setup, find_packages

setup(name='spatialpandas',
      packages=find_packages(exclude=('tests',)),
      install_requires=['pandas', 'dask', 'numba', 'numpy'],
      tests_require=['pytest', 'hypothesis'])

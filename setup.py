from setuptools import setup, find_packages

setup(name='spatialpandas',
      packages=find_packages(exclude=('tests',)),
      install_requires=['pandas', 'dask', 'numba', 'numpy', 'pyarrow>=0.15'],
      tests_require=['pytest', 'hypothesis', 'hilbertcurve', 'shapely', 'scipy'])

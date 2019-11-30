from setuptools import setup, find_packages

setup(name='spatialpandas',
      packages=find_packages(exclude=('tests',)),
      install_requires=['pandas>=0.25', 'dask>=2.0', 'numba', 'numpy', 'pyarrow>=0.15'],
      tests_require=[
            'pytest', 'hypothesis', 'scipy', 'shapely', 'geopandas', 'hilbertcurve'
      ])

from setuptools import setup, find_packages
import param

extras_require = {
      'tests': [
            'pytest',
            'hypothesis',
            'scipy',
            'shapely',
            'geopandas',
            'hilbertcurve'
      ],
}

install_requires = [
      'pandas>=0.25',
      'dask>=2.0',
      'numba',
      'numpy',
      'pyarrow>=0.15',
      'param',
]

setup(name='spatialpandas',
      version=param.version.get_setup_version(
            __file__, "spatialpandas", archive_commit="$Format:%h$"
      ),
      packages=find_packages(exclude=('tests',)),
      python_requires='>=3.6',
      install_requires=install_requires,
      extras_require=extras_require,
      tests_require=extras_require['tests'])

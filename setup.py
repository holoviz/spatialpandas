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
      'examples': [
            'geopandas',
            'matplotlib',
            'descartes',
            'datashader',
            'holoviews',
      ]
}

install_requires = [
      'pandas>=0.25',
      'dask>=2.0',
      'numba',
      'numpy',
      'pyarrow>=0.15',
      'param',
      'fsspec',
      'retrying',
]

setup(name='spatialpandas',
      version=param.version.get_setup_version(
            __file__, "spatialpandas", archive_commit="$Format:%h$"
      ),
      description='Pandas extension arrays for spatial/geometric operations',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url='https://github.com/holoviz/spatialpandas',
      maintainer='Datashader developers',
      maintainer_email='dev@datashader.org',
      python_requires='>=3.6',
      install_requires=install_requires,
      extras_require=extras_require,
      tests_require=extras_require['tests'],
      license='BSD-2-Clause',
      packages=find_packages(exclude=('tests', 'tests.*')),
      include_package_data=True)

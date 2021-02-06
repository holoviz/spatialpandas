import sys

import param
from setuptools import find_packages, setup

extras_require = {
    'tests': [
        'pytest',
        'codecov',
        'pytest-cov',
        'flake8',
        'hypothesis',
        'scipy',
        'shapely',
        'geopandas',
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
    'dask[complete] >=2.0',
    'numba',
    'pyarrow>=0.15',
    'param',
    'fsspec',
    'retrying',
]

if sys.platform == 'darwin':
    install_requires.extend(['numpy<1.20'])
else:
    install_requires.extend(['numpy'])

setup_args = dict(
    name='spatialpandas',
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
    include_package_data=True
)

if __name__ == '__main__':
    setup(**setup_args)

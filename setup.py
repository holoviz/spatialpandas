import sys

import param
from setuptools import find_packages, setup

extras_require = {
    'tests': [
        'codecov',
        'flake8',
        'geopandas',
        'hypothesis',
        'pytest-cov',
        'pytest',
        'scipy',
        'shapely',
    ],
    'examples': [
        'datashader',
        'descartes',
        'geopandas',
        'holoviews',
        'matplotlib',
    ]
}

install_requires = [
    'dask[complete] >=2.0',
    'fsspec',
    'numba',
    'pandas>=0.25',
    'param',
    'retrying',
]

if sys.platform == 'darwin':
    install_requires.extend([
        'numpy<1.20',
        'pyarrow>=0.15,<2',
    ])
else:
    install_requires.extend([
        'numpy>=1.20',
        'pyarrow>=3',
    ])

setup_args = dict(
    name='spatialpandas',
    version=param.version.get_setup_version(
        __file__,
        "spatialpandas",
        archive_commit="$Format:%h$",
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
    include_package_data=True,
)

if __name__ == '__main__':
    setup(**setup_args)

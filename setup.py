import param

from setuptools import find_namespace_packages, setup

extras_require = {
    'tests': [
        'codecov',
        'flake8',
        'hilbertcurve',
        'geopandas',
        'hypothesis',
        'keyring',
        'moto[s3,server]',
        'pytest-cov',
        'pytest',
        'python-snappy',
        'rfc3986',
        's3fs',
        'scipy',
        'shapely',
        'twine',
    ],
    'examples': [
        'datashader',
        'distributed',
        'descartes',
        'geopandas',
        'holoviews',
        'matplotlib',
    ]
}

install_requires = [
    'dask',
    'fsspec',
    'numba',
    'pandas',
    'param',
    'pyarrow >=1.0',
    'retrying',
]

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
    maintainer='HoloViz developers',
    maintainer_email='developers@holoviz.org',
    python_requires='>=3.9',
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    license='BSD-2-Clause',
    packages=find_namespace_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
)

if __name__ == '__main__':
    setup(**setup_args)

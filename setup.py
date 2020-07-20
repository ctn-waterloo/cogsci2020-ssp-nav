from setuptools import find_packages, setup

setup(
    name="ssp_navigation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'nengo',
        'numpy',
        'matplotlib',
        'seaborn',
        'pandas',
        # 'gridworlds',
        # 'spatial_semantic_pointers',
    ]
)

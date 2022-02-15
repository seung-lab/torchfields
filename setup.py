from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torchfields',
    version='0.1.2',
    author='Barak Nehoran, Nico Kemnitz',
    author_email='bnehoran@users.noreply.github.com, nkemnitz@users.noreply.github.com',
    description='A PyTorch add-on for working with image mappings and displacement fields, including Spatial Transformers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/seung-lab/torchfields",
    setup_requires=[
        'pbr',
    ],
    pbr=True,
)

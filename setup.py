from setuptools import setup, find_packages

setup(
    name='headinfer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],  # or load from requirements.txt
    author='Luo et al.',
    description='HeadInfer: inference with less gpu consumption',
    url='https://github.com/wdlctc/headinfer',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)


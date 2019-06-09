import os
from setuptools import find_packages, setup


_DIRECTORY = os.path.dirname(__file__)


with open(os.path.join(_DIRECTORY, 'README.md'), 'r', encoding='utf-8') as src:
    long_description = src.read()


with open(os.path.join(_DIRECTORY, 'requirements.txt'), 'r', encoding='utf-8') as src:
    requirements = [line
                    for line in src
                    if not line.startswith('#')]


setup(
    name='keras_elmo_bert',
    version='0.0.1',
    description='Implementation of ELMO/BERT as finetuneable keras layers',
    long_description=long_description,
    url='https://github.com/igeti/keras_elmo_bert',
    #author,
    #author_email,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],
    install_requires=requirements,
    packages=find_packages(),
    python_requires='>=3.0',
)
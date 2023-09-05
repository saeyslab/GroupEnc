
from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Encoder with group loss implemented with TensorFlow'
LONG_DESCRIPTION = 'Encoder with group loss implemented with TensorFlow for global structure preservation'

setup(
        name='SGroupVAE', 
        version=VERSION,
        author="David Novak",
        author_email="<davidnovakcz@hotmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'SQuadMDS', 'VAE', 'Encoder'],
        classifiers= [
            "Programming Language :: Python :: 3"
        ]
)

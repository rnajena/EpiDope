from setuptools import setup
import versioneer

requirements = [
    'tensorflow', 'bokeh', 'numpy', 'sklearn', 'keras', 'torch', 'allennlp'  # package requirements go here
]

setup(
    name='EpiDope',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Prediction of B-cell epitopes from amino acid sequences using deep neural networks. ",
    license="MIT",
    author="Florian Mock, Maximilian Collatz",
    author_email='florian.mock@uni-jena.de',
    url='https://github.com/flomock/EpiDope',
    packages=['epidope'],
    entry_points={
        'console_scripts': [
            'epidope=epidope.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='EpiDope',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

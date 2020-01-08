from setuptools import setup, find_packages
import versioneer

requirements = [
    'tensorflow', 'bokeh', 'numpy', 'scikit-learn', 'keras', 'torch', 'allennlp'  # package requirements go here, sklearn
]

setup(
    name='EpiDope',
    version=0.1, #versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    description="Prediction of B-cell epitopes from amino acid sequences using deep neural networks. ",
    license="MIT",
    author="Florian Mock, Maximilian Collatz",
    author_email='florian.mock@uni-jena.de',
    url='https://github.com/flomock/EpiDope',
    py_modules=['epidope'],
    packages=find_packages(),
    package_data={'epidope': ['elmo_settings/options.json','elmo_settings/weights.hdf5']},
    entry_points={
        'console_scripts': [
            'epidope=epidope.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='EpiDope',
    # classifiers=[
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    # ]
)

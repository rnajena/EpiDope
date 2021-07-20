from setuptools import setup, find_packages

requirements = [
    'tensorflow', 'bokeh', 'numpy', 'scikit-learn', 'keras', 'torch', 'allennlp'
]

setup(
    name='EpiDope',
    version=0.2,
    description="Prediction of B-cell epitopes from amino acid sequences using deep neural networks. ",
    license="MIT",
    author="Florian Mock, Maximilian Collatz",
    author_email='florian.mock@uni-jena.de',
    url='https://github.com/flomock/EpiDope',
    py_modules=['epidope'],
    packages=find_packages(),
    package_data={'epidope': ['elmo_settings/options.json', 'elmo_settings/weights.hdf5',
                              'epidope_weights/weights_model_k-fold_run_1_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_2_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_3_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_4_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_5_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_6_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_7_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_8_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_9_both_embeddings_50epochs.hdf5',
                              'epidope_weights/weights_model_k-fold_run_10_both_embeddings_50epochs.hdf5']},
    entry_points={
        'console_scripts': [
            'epidope=epidope.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='EpiDope',
    python_requires="3.6"
)

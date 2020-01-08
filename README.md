# EpiDope
Prediction of B-cell epitopes from amino acid sequences using deep neural networks.
## Installation

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.7

    ```bash
    conda create -n epidope python=3.7
    ```
    
3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use epidope.

    ```bash
    conda activate epidope
    ```
4. Install epidope via conda 

    ```bash
    conda install -c flomock -c conda-forge -c pytorch epidope
    ```
5. Install other dependencies

    ```bash
    pip install allennlp
    ```
    
## Usage
**Example**

```bash
epidope -i /path_to/multifasta.fa -o ./results/ -t 0.8 -e /known/epitopes.txt
```

**Options:**


command | what it does
  ------------- | -------------
-i, --infile          |Multi- or Singe- Fasta file with protein sequences.  [required]
-o, --outdir          |Specifies output directory. Default = .
--delim               |Delimiter char for fasta header. Default = White space
--idpos               |Position of gene ID in fasta header. Zero based. Default = 0
-t, --threshold       |Threshold for epitope score. Default = 0.818
-l, --slicelen        |Length of the sliced predicted epitopes. Default = 15
-s, --slice_shiftsize |Shiftsize of the slices on predited epitopes. Default = 5
-p, --processes       |Number of processes used for predictions. Default = #CPU-cores
-e, --epitopes        |File containing a list of known epitope sequences for plotting
-n, --nonepitopes     |File containing a list of non epitope sequences for plotting
-h, --help            |show this message and exit

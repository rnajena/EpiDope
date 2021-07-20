# EpiDope
Prediction of B-cell epitopes from amino acid sequences using deep neural networks. Supported on Linux and Mac.

## System-requirements
8 GB RAM should be available. With 8GB even processing protein sequences longer than 6000 amino acids and/or multiple hundreds of sequences shouldn't be problematic.

## Installation

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Install mamba in your conda base repository

    ```bash
    conda install -c conda-forge mamba
    ```
3. Create new repository and install epidope via mamba 

    ```bash
    mamba create -n epidope -c flomock -c conda-forge epidope
    ```
   Note: While installation the loading bar of EpiDope may not work. So depending on your internet connection, it can take from a few seconds too minutes until you see any progress.
    
4.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use epidope.

    ```bash
    conda activate epidope
    ```
   

## Usage
**Example**

```bash
epidope -i /path_to/multifasta.fa -o ./results/ -e /known/epitopes.txt
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

## Docker
We also provide a Docker image for EpiDope.  
Simply pull and run a ready-to-use image from Dockerhub:  
```bash
docker run -t --rm -v /path/to/input/files:/in -v /path/to/output:/out \
flomock/epidope:v0.2 -i /in/proteins.fasta -o /out/epidope_results
```
(you need to mount files/folders that you want to access in the Docker via `-v`)

Or if you want you can build the image yourself locally from the `Dockerfile` in this repo:
```bash
docker build -t epidope .
```

#### Note:  
Run as non-root user under linux:
```bash
docker run -t --rm -v /path/to/input/files:/in -v /path/to/output:/out -u `id -u $USER`:`id -g $USER` \
flomock/epidope:v0.2 -i /in/proteins.fasta -o /out/epidope_results
```

Run docker with a different memory allocation see [System requirements](#System-requirements) (default is 2GB for linux and mac):  
(e.g. 8GB)
```bash
docker run -t --rm -v -m=8g /path/to/input/files:/in -v /path/to/output:/out \
flomock/epidope:v0.2 -i /in/proteins.fasta -o /out/epidope_results
```

#### Further
If you are interested, you find most of the code which was used to create this tool under:

https://github.com/flomock/epitop_pred

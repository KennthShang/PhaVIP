<img src='logo.png'>

PhaVP is a python library for phage protein annotation.  It has two functions. First, it can classify a protein into either phage virion protein (PVPs) or non-PVPs (binary classification task). Second, it can assign a more detailed annotation for predicted PVPs, such as major capsid, major tail, and portal (multi-class classification task).


PhaVP is based on chaos game representation and Vision Transformer model. This GitHub is the local version of PhaVP. The webserver of PhaVP is avaliable via [server version](https://phage.ee.cityu.edu.hk/phavp). 

In addition, we provide many other phage-related analysis tools, such as [phage identification](https://github.com/KennthShang/PhaMer), [taxonomy classification](https://github.com/KennthShang/PhaGCN), [lifestyle prediction](https://github.com/KennthShang/PhaTYP), and [host prediction](https://github.com/KennthShang/CHERRY). Feel free to check them out on our website [PhaBOX](https://phage.ee.cityu.edu.hk/). 

# Overview


## Required Dependencies
Detailed package information can be found in `phavp.yaml`

If you want to use the gpu to accelerate the program please install the packages below:
* cuda
* Pytorch-gpu

Search [pytorch](https://pytorch.org/) to find the correct cuda version based on your computer.


## Quick install
*Note*: we suggest you to install all the package using conda (both [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Anaconda](https://anaconda.org/) are ok).

After cloning this respository, you can use anaconda to install the **phavp.yml**. This will install all packages you need with cpu mode. The command is: `conda env create -f phavp.yml -n phavp`

Once installed, you only need to activate your 'phavp' environment before using phavp in the next time.
```
conda activate phabox
```

## Usage 

### Run all pipelines in one command:

```
python run_PhaVP.py [--filein INPUT_FA] [--threads NUM_THREAD] [--type IN_TYPE] [--task TASK] [--tool TOOL_PTH] [--root ROOT_PTH] [--midfolder MID_PTH] [--out OUTPUT_PTH] 
```


**Options**


      --filein INPUT_FA
                            The path of your input fasta file.
      --threads NUM_THREAD
                            Number of threads to run PhaMer (default 8)
      --type IN_TYPE
                            Input type of the fasta: protein or dna (default protein)  
      --task TASK
                            Task: binary or multi (default binary)  
      --tool TOOL_PTH
                            The folder where you downlod PhaVP (default PhaVP/)
      --root ROOT_PTH
                            The folder you want to store the outputs of PhaVP (default user_0/)
      --out OUTPUT_PTH
                            The output folder under root. All the prediction will be stored in this folder. (default out/)
      --midfolder MID_PTH
                            The midfolder under root. All the intermediate files will be stored in this folder. (default midfolder/)


**Example**


Prediction on the example file under PhaVP folder:

    python run_PhaVP.py --contigs test_contigs.fa --threads 8 --type dna --task binary --tool ./ --root sample_test/ --midfolder midfolder/ --out out/
    
    
Prediction on your data file out of PhaVP folder:

    python run_PhaVP.py --contigs PATH/TO/FASTA/test_contigs.fa --threads 8 --type dna --task binary --tool PATH/TO/PhaVP --root ~/user_0/ --midfolder midfolder/ --out out/
    
### References
Not available yet.

### Contact
If you have any questions, please email us: jyshang2-c@my.cityu.edu.hk




# Controlled Comparison of Encoding Schemes for Peptide-HLA Based CD8+ T Cell Immunogenicity Prediction

## About The Project
Immunogenicity prediction is the prediction of whether a peptide is capable of triggering the activation of immune system cells. This project focuses on a specific type of immune system cell, CD8+ T cells, which bind antigen-HLA class I complexes on the surface of nucleated cells in humans. Predicting immunogenicity in this context has applications in vaccine design and CAR T-cell cancer therapy. 

The peptide amino acid sequence must be converted into a numeric format to be used as input for machine learning models, a process called encoding. This project aims to compare encoding schemes from popular paradigms in a controlled manner to identify their relative performance benefits. This addresses the lack of research on which encoding schemes maximise predictive performance.
<br>
<br>

## Repository Status
> **Disclaimer:** This repository contains experimental research code. The codebase has not yet been refactored for production.

<br>

## Requirements
A Python virtual environment can be set up with the following commands:
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

> **Note:** Structure-based encoding schemes (located in `encoders/myImmunoStruct` and `encoders/gnn.py`) are currently incomplete and may require alternative environment configurations

<br>

## Usage 
1. Encoded datasets can be generated with:
```bash
python -m benchmark.encoders.<ESM | AAindex_pca | OHE>
```

2. The controlled comparison can be run with `Project/` as the working directory with:
```bash
python -m benchmark.main [options]
```
> **Note:** Requires that encoded datasets have been generated (see previous step)

The behaviour of `main` can be configured via command line arguments:
```
usage: main.py [-h] [-o | -s SAVED] [-p PARAMSF] [-r RESULTSF] [-t TESTF]
               [--epochs EPOCHS] [--repeats REPEATS] [-B B] [-n N]

options:
  -h, --help            show this help message and exit
  -o, --optim           optimise model hyperparameters
  -s SAVED, --saved SAVED
                        pickle file with saved hyperparamters
  -p PARAMSF, --paramsf PARAMSF
                        pickle file to save model hyperparameters. If --optim
                        then specify file with saved model parameters
  -r RESULTSF, --resultsf RESULTSF
                        pickle file to store benchmark results on IEDB dataset
  -t TESTF, --testf TESTF
                        pickle file to store benchmark results on test
                        datasets
  --epochs EPOCHS       number of training epochs
  --repeats REPEATS     number of repeats
  -B B                  number of bootstrap iterations
  -n N                  fraction of dataset to resample in bootstrap
                        resampling
```

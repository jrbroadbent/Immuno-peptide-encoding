# Controlled Comparison of Encoding Schemes for Peptide-HLA Based CD8+ T Cell Immunogenicity Prediction

## About The Project
<table>
  <tr>
    <td width="60%" valign="top">
      <p>Immunogenicity prediction is the prediction of whether a peptide is capable of triggering the activation of immune system cells (see right schematic). This project focuses on a specific type of immune system cell, CD8+ T cells, which bind antigen-HLA class I complexes on the surface of nucleated cells in humans. Predicting immunogenicity in this context has applications in vaccine design and CAR T-cell cancer therapy.</p>

<p>The peptide amino acid sequence must be converted into a numeric format to be used as input for machine learning models, a process called encoding. This project aims to compare encoding schemes from popular paradigms in a controlled manner to identify their relative performance benefits. This addresses the lack of research on which encoding schemes maximise predictive performance.</p>

<br>
    
<img width="100%" alt="Mean Bootstrap binary cross-entropy (BCE) losses for ESM, AAindex + PCA, AAindex, and OHE models for A) SARS-CoV-2, B) Dengue virus, and C) Neoantigen test datasets. Bars indicate 95% confidence intervals." src="https://github.com/user-attachments/assets/e4b75b31-8851-481e-bbdf-3379fe8ddeef" />
<b>Figure:</b> Mean Bootstrap binary cross-entropy (BCE) losses for ESM, AAindex + PCA, AAindex, and OHE models for A) SARS-CoV-2, B) Dengue virus, and C) Neoantigen test datasets. Bars indicate 95% confidence intervals.
        
  </td>
  <td width="40%" valign="top">
      <img width="100%" alt="schematic of CD8+ T cell activation" src="https://github.com/user-attachments/assets/5f773fc8-fcca-4f6b-afd1-86b32af6aa6b" align="right">
    </td>
  </tr>
</table>

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

Alternatively, using docker: 
```bash 
docker pull jrboadbent/immuno-image   # pull image from docker hub
docker container run -it jrbroadbent/immuno-image /bin/bash   # run container with bash shell
```
```bash
exit   # stop the container when done
```

> **Note:** Structure-based encoding schemes (located in `benchmark/encoders/adapted_ImmunoStruct` and `benchmark/encoders/gnn.py`) are currently incomplete and may require alternative environment configurations

<br>

## Usage 
1. Clone the repository locally:
```bash
git clone https://github.com/jrbroadbent/Project.git Immuno-peptide-encoding/
cd Immuno-peptide-encoding/
```

2. Encoded datasets can be generated with:
```bash
python -m benchmark.encoders.OHE
python -m benchmark.encoders.AAindex_pca
python -m benchmark.encoders.ESM
```

3. The controlled comparison can be run with:
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

<br>

## Tests
tests can be run with pytest using:
```bash 
pip install pytest
pytest benchmark/tests/test_main.py
pytest benchmark/tests/test_model_defs.py
```

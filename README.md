# Controlled Comparison of Encoding Schemes for Peptide-HLA Based CD8+ T Cell Immunogenicity Prediction

## About The Project
<table>
  <tr>
    <td width="60%" valign="top">
      <p>The identification of immunogenic peptides is crucial for the development of vaccines and cancer immunotherapies. Peptides are said to be immunogenic if they can activate immune system cells, such as T cells (see right schematic). However, traditional screening for peptides that activate CD8+ T cells is a time-consuming and labour-intensive process. Machine learning methods are a promising complement to reduce the experimental burden and are increasingly being adopted into immunogenicity screening workflows. A critical component of machine learning models for immunogenicity prediction is the choice of encoding scheme. This determines how peptide sequences are represented and therefore what information about the peptides is available to the model which ultimately shapes model performance. However, current immunogenicity predictors deploy a range of encoding schemes, and it is unclear which one best captures the important features of immunogenic peptides to improve predictive capacity.</p>
      <p>This study addresses the lack of guidance on encoding schemes by conducting a controlled comparison of encoding schemes from key paradigms, isolating their contribution to performance to help inform the choice of encoding scheme for immunogenicity predictors. This study found that physiochemical property based encoding schemes (AAindex and AAindex+PCA) were capable of outperforming a more
advanced learned representation approach (ESM), and one-hot encoding (OHE) resulted in substantial performance variability but displayed surprising generalisation potential. See the figure below for encoding scheme performance when tested on three diverse datasets.</p>

<br>
    
<img width="100%" alt="Mean Bootstrap binary cross-entropy (BCE) losses for ESM, AAindex + PCA, AAindex, and OHE models for A) SARS-CoV-2, B) Dengue virus, and C) Neoantigen test datasets. Bars indicate 95% confidence intervals." src="https://github.com/user-attachments/assets/e4b75b31-8851-481e-bbdf-3379fe8ddeef" />
<b>Figure:</b> Mean Bootstrap binary cross-entropy (BCE) losses for ESM, AAindex + PCA, AAindex, and OHE models for <b>A)</b> SARS-CoV-2, <b>B)</b> Dengue virus, and <b>C)</b> Neoantigen test datasets. Bars indicate 95% confidence intervals.
        
  </td>
  <td width="40%" valign="top">
      <img width="100%" alt="schematic of CD8+ T cell activation" src="https://github.com/user-attachments/assets/5f773fc8-fcca-4f6b-afd1-86b32af6aa6b" align="right"\>
      <b>Figure:</b> CD8+ T cell activation. 
    </td>
  </tr>
</table>

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
docker build -t immuno-image .
docker container run -it immuno-image /bin/bash
```
This runs the container with a bash shell.
Use `exit` to stop the container when done.

<br>

## Usage 
1. Clone the repository locally:
```bash
git clone https://github.com/jrbroadbent/Immuno-peptide-encoding.git
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

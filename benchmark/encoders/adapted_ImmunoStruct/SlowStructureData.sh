#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

# [CPU] Step 1-3. Prepare MSA for AlphaFold.
# Download colabfold and REMEMBER where it is downloaded to. Likely default to `~/.cache/colabfold`.
# python -m colabfold.download

# Prepare MSA.
conda activate immuno
#cd immunostruct/preprocessing
python step1-3_server_sequence_to_msa.py \
    --input-csv ../../data/remove0123_sample100.csv \
    --output-dir ../../data/pdb_files/IEDB/ \
    --tmp-dir /tmp/ \
    --allele-sequence-csv ../../data/HLA_allele_sequences.csv\
    --allele-col-name HLA \
    --peptide-col-name peptide

# [GPU] Step 4. AlphaFold2.
# start/end help run multiple jobs in parallel.
python step4_msa_to_pdb.py \
    --input-csv ../../data/remove0123_sample100.csv \
    --output-dir ../../data/pdb_files/IEDB/ \
    --start 0 --end 24540 \
    --params-loc ~/.cache/colabfold \
    --allele-col-name allele \
    --peptide-col-name peptide

# [CPU] Step 5. Moving and renaming the structure data in PDB files.
python step5_rename_pdb.py \
    --input-dir ../../data/pdb_files/IEDB/ \
    --output-dir ../../data/alphafold2_pdb_IEDB/

# [CPU] Step 6. Generating PyG graphs (structures in PDB files to structures in PyTorch .pt files).
python step6_pdb_to_pyg.py \
    --input-dir ../../data/alphafold2_pdb_IEDB/ \
    --output-dir ../../data/graph_pyg_IEDB/


2026-06-19 13:58:21.251654: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/lib

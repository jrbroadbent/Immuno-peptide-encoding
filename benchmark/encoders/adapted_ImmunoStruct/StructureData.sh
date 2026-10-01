#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

# [CPU] Step 1-3. Prepare MSA for AlphaFold.
# Download colabfold (AlphaFold2 weights) and REMEMBER where it is downloaded to. Likely default to `~/.cache/colabfold`.
# python -m colabfold.download  # run in immuno conda env --> downloads AlphaFold2 weights

# Download the MSA database locally.
mkdir ./database_msa/
cd ./database_msa/
wget https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
tar -xzvf uniref30_2302.tar.gz
mmseqs tsv2exprofiledb uniref30_2302 uniref30_2302_db
wget https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz
tar -xzvf colabfold_envdb_202108.tar.gz
mmseqs tsv2exprofiledb colabfold_envdb_202108 colabfold_envdb_202108_db
cd ..

# Prepare MSA.
conda activate local_msa
# cd immunostruct/preprocessing
python step1_local_sequence_to_fasta.py \
    --input-csv ../../data/remove0123_sample100.csv \  # changed data file name
    --output-fasta ../../data/fasta/ImmunoStruct_IEDB_data.fasta \
    --allele-col-name allele \
    --peptide-col-name peptide
python step2_local_fasta_to_a3m.py \
    --input-fasta ../../data/fasta/ImmunoStruct_IEDB_data.fasta \
    --msa-database-dir ../../database_msa/ \
    --output-dir ../../data/a3m/IEDB/
python step3_local_a3m_to_msa.py \
    --input-dir ../../data/a3m/IEDB/ \
    --output-dir ../../data/pdb_files/IEDB/ \
    --input-csv ../../data/ImmunoStruct_IEDB_data.csv \
    --allele-col-name allele \
    --peptide-col-name peptide

# [GPU] Step 4. AlphaFold2.
# start/end help run multiple jobs in parallel.
conda deactivate
conda activate immuno
python step4_msa_to_pdb.py \
    --input-csv ../../data/remove0123_sample100.csv \  # changed data file name
    --output-dir ../../data/pdb_files/IEDB/ \
    --start 0 --end 24540 \  # what are these numbers??
    --params-loc ~/.cache/colabfold \  # input path to colabfold (downloaded AlphaFold2 weights)
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

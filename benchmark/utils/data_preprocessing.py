"""
Data collection and preprocessing utility functions:
Query IEDB database
group dataset entries based on mmseqs2 clustering or parent proteins
requires conda environment for mmseqs2 (conda activate proj_venv)
how did I create the conda environment? use pip inside active conda env to install requiremetns and also install mmseqs2 (using commands on GitHub)
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path
import subprocess
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).resolve().parent / 'tcell_assays_iedb.csv'


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise

def query_tcell_epitope():
    base_uri='https://query-api.iedb.org'
    table_name='tcell_search'
    full_url=f"{base_uri}/{table_name}"
    search_params={
        "order": "tcell_id",
        "structure_type": "eq.Linear peptide",
        "mhc_class": "eq.I",
        "host_organism_iri_search": "ov.{NCBITaxon:9606}" # human
        }
    headers={"Prefer": "count=exact"}
   
    page_size = 10000
    offset = 0
    frames = []

    with requests.Session() as s:
        while True:
            r = s.get(full_url, params={**search_params, "offset": str(offset), "limit": str(page_size)}, headers=headers)
            check_response(r)

            batch = pd.json_normalize(r.json())

            if batch.empty:
                break

            # print GET progress:
            print(f"offset: {offset}. Query returned {len(batch)} entries")

            frames.append(batch)

            if len(batch) < page_size:
                break

            offset += page_size

    df = pd.concat(frames, ignore_index=True)

    df.to_csv(OUTPUT_PATH)
    print(f"\nsaved to {OUTPUT_PATH}")
    
    return 0


def query_uniprot(uniprot_ids, poll_interval=2):
    API = "https://rest.uniprot.org"
    session = requests.Session()

    # submit job 
    r = session.post(
        f"{API}/idmapping/run",
        data={
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ",".join(uniprot_ids),
        },
    )
    r.raise_for_status()
    job_id = r.json()["jobId"]
    print(f"Submitted job {job_id}")

    # poll
    while True:
        r = session.get(f"{API}/idmapping/status/{job_id}")
        r.raise_for_status()
        status = r.json().get("jobStatus")

        if status in ("NEW", "RUNNING"):
            print("Still running...")
            time.sleep(poll_interval)
        elif status in (None, "FINISHED"):
            break
        else:
            raise RuntimeError(f"Job failed with status: {status}")

    # stream results 
    r = session.get(
        f"{API}/idmapping/uniprotkb/results/stream/{job_id}",
        params={
            "format": "tsv",
            "fields": "accession,sequence",
        },
    )
    r.raise_for_status()

    sequences = {}
    for line in r.text.strip().splitlines()[1:]:  # skip header row
        original_id, _, seq = line.split("\t")
        sequences[original_id] = seq

    missing = set(uniprot_ids) - set(sequences)
    if missing:
        print(f"Warning: {len(missing)} IDs returned no sequence: {sorted(missing)[:10]}")

    return sequences


def get_parents():
    parents = {}

    return parents 


def create_fasta(sequences):
    with open("parents.fasta", "wt") as f:
        for sequence in sequences:
            f.write(f">{sequence}\n{sequences[sequence]}\n")
    return 0


def group_by_seqid(cluster_file):
    group_id = 0
    groups = {}
    with open(cluster_file, "rt") as f:
        for line in f:
            rep, mem = line.strip().split("\t")
            if rep == mem:
                group_id += 1
            groups[mem] = group_id
    return groups


def process_parent_iri(dataset_iris):
    parent_iris = []
    for iri in dataset_iris:
        if type(iri) == str and ":" in iri and "other" not in iri:
            identifier = iri.split(sep=":")[1]
        else:
            identifier = None
        parent_iris.append(identifier)
    return parent_iris



if __name__ == '__main__':
    # get dataset
    query_tcell_epitope()

    dataset = pd.read_csv(OUTPUT_PATH,
                          usecols = ["linear_sequence", 
                                      "linear_sequence_length",
                                      "mhc_restriction", 
                                      "mhc_allele_resolution", 
                                      "parent_source_antigen_iri", 
                                      "assay_description"],
                          dtype = {"linear_sequence": str,
                                   "linear_sequence_length": int,
                                    "mhc_restriction": str, 
                                    "mhc_allele_resolution": str, 
                                    "parent_source_antigen_iri": str, 
                                    "assay_description": str})


    #### Filter #### 
    # how many entries with NA values have I got? significant proportion? what to do with them? 97 in mhc_allele_resolution
    # remove repeats (same peptide-HLA pair)
    # drop entries with NAs in mhc_restriction or assay_description (shouldn't be any) 
    # by linear sequence length (only 9 and 10)
    # by mhc allele resolution? (only take 2 chain resolution?)
    
    print(f"size of dataset: {dataset.shape}\n{dataset.head()}\nMHC allele resolution column:\n{dataset["mhc_allele_resolution"]}")
    print(f"number of NAs:\nmhc restriction: {dataset["mhc_restriction"].isna().sum()}\nmhc allele resolution: {dataset["mhc_allele_resolution"].isna().sum()}\nassay description: {dataset["assay_description"].isna().sum()}")
    dataset = dataset[dataset["mhc_allele_resolution"].str.contains(r"(\b2\s+chain)", case=False, na=False)]

    dataset = dataset[dataset["linear_sequence_length"].isin([9, 10])]

    # check unique values of columns; e.g. mhc_allele_resolution col should only contain "2 chain"
    print(f"mch allele resolution unique values: {dataset["mhc_allele_resolution"].unique()}")
    print(f"sequence length unique values: {dataset["linear_sequence_length"].unique()}")
    print(f"shape of dataset: {dataset.shape}") # --> lost half of the dataset from the filtering (~40%) - yikes!!


    # process assay description column - if contains positive assign "Positive" if contains negative assign "Negative"
    dataset["immunogenicity"] = dataset["assay_description"].str.extract(r"([Pp]ositive|[Nn]egative)", expand=False)
    print(dataset["immunogenicity"].unique()) # contains only Positive or Negative --> no NAs

    # remove repeat peptide-HLA pairs
    # is unique? --> no; parent iri and immunogenicity the same across repeats? --> no; is one more common? --> no; remove all instances 
    pep_hla_pair = dataset.loc[["linear_sequence","mhc_restriction"]]
    unique_pairs = set(pep_hla_pair)

    for (peptide, hla) in pep_hla_pair:
        if (peptide, hla) in unique_pairs:


    dataset = dataset.set_index(pd.Index(range(dataset.shape[0])))

    print(dataset.head(),"\n", dataset.shape)
    

    # create set of parent protein uniprot iris
    dataset_iris = dataset["parent_source_antigen_iri"]
    non_unique_iris = process_parent_iri(dataset_iris)
    parent_iris = set(non_unique_iris)
    parent_iris.discard(None)
   

    # get parent protein sequences from uniprot
    sequences = query_uniprot(parent_iris)
    print(f"{len(parent_iris)}\n{len(sequences)}")

    # create fasta file
    create_fasta(sequences)

    # cluster parent proteins with mmseqs2
    subprocess.run(["mmseqs", "easy-cluster", "parents.fasta", "clusterRes", "tmp", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0"])
    groups_dic = group_by_seqid("clusterRes_cluster.tsv")

    # Create list for group Ids
    no_data_count = 0
    groups = []
    extra_group = len(groups_dic)+1
    for iri in non_unique_iris:
        if iri in groups_dic:
            groups.append(groups_dic[iri])
        else: 
            groups.append(extra_group)
            extra_group += 1
            no_data_count += 1

    print(f"percentage of entries with no parent protein info: {round(no_data_count/len(groups)*100, 1)}%")

    # inspect distribution of group sizes
    group_sizes = []
    for group in set(groups):
        size = groups.count(group)
        group_sizes.append(size)
    group_sizes.sort(reverse=True)
    print(f"number of groups: {len(group_sizes)} = {len(set(groups))}, number of entries: {len(non_unique_iris)} = {len(dataset_iris)}")

    # plot histogram 
    plt.hist(group_sizes, bins=20, density=True, log=True)
    plt.ylabel("log(density)")
    plt.xlabel("Group size")
    plt.savefig("GroupSizeDist.png")


    # drop unnecessary columns 
    # rename columns to match structure
    dataset["group_ids"] = groups
    dataset = dataset.drop(["linear_sequence_length",
                    "mhc_allele_resolution", 
                    "parent_source_antigen_iri", 
                    "assay_description"], 
                    axis="columns")

    dataset = dataset.rename(columns={"linear_sequence": "peptide",
                            "mhc_restriction": "HLA"})
                                      
    dataset.to_csv("2026_iedb_data.csv", header=True)

    # create splits
    # StratifiedGroupKFold()


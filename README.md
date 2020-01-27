# Locus-to-gene scoring pipeline

Workflows to run locus-to-gene (L2G) ML scoring pipeline. This repository conatins code to:

1. Engineer L2G feature-matrix
2. Process and join gold-standards (training data)
3. Train XGBoost classifier
4. Validate model under cross-validation
5. Use model to prioritise genes for all loci

## Setup environment

```bash
# Install java 8 e.g. using apt
sudo apt install -yf openjdk-8-jre-headless openjdk-8-jdk

# Authenticate google cloud storage
gcloud auth application-default login

# Install dependencies into isolated environment
conda env create -n l2g --file environment.yaml

# Activate environment, set RAM availability and output version name
conda activate l2g
export PYSPARK_SUBMIT_ARGS="--driver-memory 50g pyspark-shell"
version_date=`date +%y%m%d`
```

## Step 1: Feature engineering

Integrates fine-mapping information with functional genomics datasets to generate L2G predictive features.

### Start Dataproc cluster

```bash
# Start cluster (highmem)
gcloud beta dataproc clusters create \
    em-cluster-ml-features \
    --image-version=1.4 \
    --properties=spark:spark.debug.maxToStringFields=100,spark:spark.executor.cores=63,spark:spark.executor.instances=1 \
    --metadata 'PIP_PACKAGES=networkx==2.1 pandas==0.25.0' \
    --initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
    --master-machine-type=n1-highmem-64 \
    --master-boot-disk-size=1TB \
    --num-master-local-ssds=1 \
    --zone=europe-west1-d \
    --initialization-action-timeout=20m \
    --single-node \
    --max-idle=10m

# To monitor
gcloud compute ssh em-cluster-ml-features-m \
  --project=open-targets-genetics \
  --zone=europe-west1-d -- -D 1080 -N

"EdApplications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --proxy-server="socks5://localhost:1080" \
  --user-data-dir="/tmp/em-cluster-ml-features-m" http://em-cluster-ml-features-m:8088
```

### Generate features

Input file paths are specified in `1_feature_engineering/1_prepare_inputs.py`!

```bash
# Change to working directory
cd 1_feature_engineering

# Update input file paths to latest versions
nano 1_prepare_inputs.py

# Create input datasets
gcloud dataproc jobs submit pyspark \
    --cluster=em-cluster-ml-features \
    1_prepare_inputs.py -- --version $version_date

# Create features.
# N.B. distance must be run last.
cd 2_make_features
for inf in dhs_prmtr.py enhc_tss.py fm_coloc.py others.py pchicJung.py pics_coloc.py polyphen.py vep.py distance.py; do
    gcloud dataproc jobs submit pyspark \
        --cluster=em-cluster-ml-features \
        $inf -- \
        --version $version_date
done

# Copy proteinAttenuation feature over (no alterations needed)
gsutil -m rsync -r \
  gs://genetics-portal-staging/l2g/$version_date/features/inputs/proteinAttenuation.parquet \
  gs://genetics-portal-staging/l2g/$version_date/features/output/separate/proteinAttenuation.parquet

# Join features
gcloud dataproc jobs submit pyspark \
    --cluster=em-cluster-ml-features \
    3_join_features.py -- --version $version_date

# Change back to root directory
cd ..
```

## Step 2: Join gold-standard training data

Processes the gold-standard training data and joins with the feature matrix.

### Prepare input data

```bash
# Change to working directory
cd 2_process_training_data

# Args
gold_standards_url=https://raw.githubusercontent.com/opentargets/genetics-gold-standards/master/gold_standards/processed/gwas_gold_standards.191108.jsonl
feature_matrix_gs=gs://genetics-portal-staging/l2g/$version_date/features/output/features.raw.$version_date.parquet

# Download input data to local machine
mkdir -p input_data
# a. Gold-standard GWAS loci
wget --directory-prefix input_data $gold_standards_url
# b. Feature matrix
gsutil -m rsync -r \
  gs://genetics-portal-staging/l2g/$version_date/features/output/features.raw.$version_date.parquet \
  input_data/features.raw.$version_date.parquet
# c. Other feature inputs (string, cluster and toploci data)
gsutil -m rsync -r \
  gs://genetics-portal-staging/l2g/$version_date/features/inputs/string.parquet \
  input_data/string.$version_date.parquet
gsutil -m rsync -r \
  gs://genetics-portal-staging/l2g/$version_date/features/inputs/clusters.parquet \
  input_data/clusters.$version_date.parquet
gsutil -m rsync -r \
  gs://genetics-portal-staging/l2g/$version_date/features/inputs/toploci.parquet \
  input_data/toploci.$version_date.parquet

# Process and join gold-standards to feature matrix
python process_goldstandards.py /
  --in_features input_data/features.raw.$version_date.parquet \
  --in_gs input_data/$(basename $gold_standards_url) \
  --in_string input_data/string.$version_date.parquet \
  --in_clusters input_data/clusters.$version_date.parquet \
  --in_toploci input_data/toploci.$version_date.parquet \
  --out_full output/featurematrix_w_goldstandards.full.$version_date.parquet \
  --out_training output/featurematrix_w_goldstandards.training_only.$version_date.parquet \
  --out_log_dir output/logs_$version_date

# Change back to root directory
cd ..
```

## Step 3: Train classifier

## Step 4: Validate model

## Step 5: Prioritise genes
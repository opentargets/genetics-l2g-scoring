# Locus-to-gene scoring pipeline

This repository contains workflows to run the Open Targets Genetics locus-to-gene (L2G) ML scoring pipeline.

> **Note:** this is not a standalone pipeline, in that it needs inputs in a particular format that are generated by our other pipelines (genetics-v2d, genetics-v2g, genetics-finemapping, genetics-colocalisation). Therefore, at present it is not practical for external users to directly run the pipeline.

Steps:
1. [Engineer L2G feature-matrix](#step-1-feature-engineering)
2. [Process and join gold-standards (training data)](#step-2-join-gold-standard-training-data)
3. [Train XGBoost classifier](#step-3-train-classifier)
4. [Validate model under cross-validation](#step-4-validate-model)
5. [Use model to prioritise genes for all loci](#step-5-prioritise-genes)

Step 1 is run on DataProc. Steps 2-5 run on local compute (e.g. 32 core standard GCP).

## Setup environment

```bash
# Install java 8 e.g. using apt
sudo apt-get update -y
sudo apt install -yf openjdk-8-jre-headless openjdk-8-jdk

# Authenticate google cloud storage
gcloud auth application-default login

# Install dependencies into isolated environment
conda env create -n l2g --file environment.yaml

# Activate environment, set RAM availability and output version name
conda activate l2g
export PYSPARK_SUBMIT_ARGS="--driver-memory 100g pyspark-shell"
version_date=`date +%y%m%d`
version_date=220212
```

## Step 1: Feature engineering

Integrates fine-mapping information with functional genomics datasets to generate L2G predictive features.

### Start Dataproc cluster

```bash
# Start cluster (highmem)
gcloud beta dataproc clusters create \
    em-cluster-ml-features \
    --image-version=1.4 \
    --region europe-west1 \
    --zone=europe-west1-d \
    --properties=spark:spark.debug.maxToStringFields=100,spark:spark.executor.cores=12,spark:spark.executor.instances=5 \
    --metadata 'PIP_PACKAGES=networkx==2.1 pandas==0.25.0' \
    --initialization-actions gs://dataproc-initialization-actions/python/pip-install.sh \
    --master-machine-type=n2-highmem-64 \
    --master-boot-disk-size=1TB \
    --num-master-local-ssds=0 \
    --initialization-action-timeout=20m \
    --single-node \
    --max-idle=10m

# To monitor
gcloud compute ssh em-cluster-ml-features-m \
  --project=open-targets-genetics-dev \
  --zone=europe-west1-d -- -D 1080 -N

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --proxy-server="socks5://localhost:1080" \
  --user-data-dir="/tmp/em-cluster-ml-features-m" http://em-cluster-ml-features-m:8088
```

### Generate features

Input file paths are specified in `1_feature_engineering/1_prepare_inputs.py`!

```bash
tmux

# Change to working directory
cd 1_feature_engineering

# Update input file paths to latest versions
nano 1_prepare_inputs.py

# Create input datasets (took 32 min on last run)
gcloud dataproc jobs submit pyspark \
    --cluster=em-cluster-ml-features \
    --region europe-west1 \
    1_prepare_inputs.py -- --version $version_date

# Create features. (~1 hr?)
# N.B. distance must be run last.
cd 2_make_features
for inf in dhs_prmtr.py enhc_tss.py fm_coloc.py others.py pchicJung.py pics_coloc.py polyphen.py vep.py distance.py; do
    gcloud dataproc jobs submit pyspark \
        --cluster=em-cluster-ml-features \
        --region europe-west1 \
        $inf -- \
        --version $version_date
done

# Copy proteinAttenuation feature over (no alterations needed)
gsutil -m rsync -r \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/inputs/proteinAttenuation.parquet \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/output/separate/proteinAttenuation.parquet

# Join features (takes a couple min)
cd ..
gcloud dataproc jobs submit pyspark \
    --cluster=em-cluster-ml-features \
    --region europe-west1 \
    3_join_features.py -- --version $version_date

# Change back to root directory
cd ..
```

## Step 2: Join gold-standard training data

Processes the gold-standard training data and joins with the feature matrix.

```bash
# Change to working directory
cd 2_process_training_data

# Args
gold_standards_url=https://raw.githubusercontent.com/opentargets/genetics-gold-standards/master/gold_standards/processed/gwas_gold_standards.191108.jsonl

# Download input data to local machine
mkdir -p input_data/features.raw.$version_date.parquet \
  input_data/string.$version_date.parquet \
  input_data/clusters.$version_date.parquet \
  input_data/toploci.$version_date.parquet
# a. Gold-standard GWAS loci
wget --directory-prefix input_data $gold_standards_url
# b. Feature matrix
gsutil -m rsync -r \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/output/features.raw.$version_date.parquet \
  input_data/features.raw.$version_date.parquet
# c. Other feature inputs (string, cluster and toploci data)
gsutil -m rsync -r \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/inputs/string.parquet \
  input_data/string.$version_date.parquet
gsutil -m rsync -r \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/inputs/clusters.parquet \
  input_data/clusters.$version_date.parquet
gsutil -m rsync -r \
  gs://genetics-portal-dev-staging/l2g/$version_date/features/inputs/toploci.parquet \
  input_data/toploci.$version_date.parquet

# Process and join gold-standards to feature matrix
python process_goldstandards.py \
  --in_features input_data/features.raw.$version_date.parquet \
  --in_gs input_data/$(basename $gold_standards_url) \
  --in_string input_data/string.$version_date.parquet \
  --in_clusters input_data/clusters.$version_date.parquet \
  --in_toploci input_data/toploci.$version_date.parquet \
  --out_full output/featurematrix_w_goldstandards.full.$version_date.parquet \
  --out_training output/featurematrix_w_goldstandards.training_only.$version_date.parquet \
  --out_log_dir output/logs_$version_date

# Backup to GCS
gsutil -m rsync -r output gs://genetics-portal-dev-staging/l2g/$version_date/gold_standards/
gsutil cp input_data/$(basename $gold_standards_url) gs://genetics-portal-dev-staging/l2g/$version_date/gold_standards/

# Change back to root directory
cd ..
```

## Step 3: Train classifier

Uses feature matrix and gold-standards to train a classifier

Note: It takes 2 hours to train all models using 500 iterations on 64 cores

```bash
# Change to working directory
cd 3_train_model

# Train model 
python train_model_xgboost.py \
  --in_path ../2_process_training_data/output/featurematrix_w_goldstandards.training_only.$version_date.parquet \
  --out_dir output/$version_date/models

# Backup to GCS
gsutil -m rsync -r output/$version_date/models gs://genetics-portal-dev-staging/l2g/$version_date/models/

# Change back to root directory
cd ..
```

## Step 4: Validate model

Takes the trained models and produces:

1. validation plots
2. classification table

```bash
# Change to working directory
cd 4_validate_model

# Load models and make predictions across training data
# (don't expand model pattern glob)
python 1_predict_training_data.py \
  --in_ft ../2_process_training_data/output/featurematrix_w_goldstandards.training_only.$version_date.parquet \
  --in_model_pattern ../3_train_model/output/$version_date/models/'*.model.joblib.gz' \
  --out_pred output/$version_date/intermediate_data/predictions_trainingonly.parquet \
  --out_ftimp output/$version_date/intermediate_data/feature_importances.json

# Generate validation plots and classification report
python 2_validation_plots.py \
  --in_pred output/$version_date/intermediate_data/predictions_trainingonly.parquet \
  --in_ftimp output/$version_date/intermediate_data/feature_importances.json \
  --out_plotdir output/$version_date/plots \
  --out_report output/$version_date/classification_report.tsv

# Backup to GCS
gsutil -m rsync -r output/$version_date gs://genetics-portal-dev-staging/l2g/$version_date/validation/

# Change back to root directory
cd ..
```

## Step 5: Prioritise genes

Prioritises genes across all loci in the feature matrix. Processes output for L2G table.

```bash
# Change to working directory
cd 5_prioritise_genes

# If the model wasn't retrained, then you could use an old version here
model_version=$version_date
model_version=220128

# Load models and make predictions across all loci
# (don't expand model pattern glob)
python 1_prioritise_genes_all_loci.py \
  --in_ft ../2_process_training_data/output/featurematrix_w_goldstandards.full.$version_date.parquet \
  --in_model_pattern ../3_train_model/output/$model_version/models/'*.model.joblib.gz' \
  --out_long output/$version_date/predictions.full.$version_date.long.parquet

# Format locus-to-gene table for export
python 2_format_l2g_table.py \
  --in_long output/$version_date/predictions.full.$version_date.long.parquet \
  --out_l2g output/$version_date/l2g.full.$version_date.parquet \
  --exclude_studies GCST007236 \
  --keep_clf xgboost \
  --keep_gsset high_medium

# Backup to GCS
gsutil -m rsync -r output/$version_date gs://genetics-portal-dev-staging/l2g/$version_date/predictions/

# Change back to root directory
cd ..
```
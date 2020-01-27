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

# Activate environment and set RAM availability
conda activate l2g
export PYSPARK_SUBMIT_ARGS="--driver-memory 50g pyspark-shell"
```

## Step 1: Feature engineering

## Step 2: Join gold-standard training data

## Step 3: Train classifier

## Step 4: Validate model

## Step 5: Prioritise genes
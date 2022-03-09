### Start Dataproc cluster

```bash
# Cluster initialization with pip
gcloud dataproc clusters create il-l2g \
    --image-version=2.0 \
    --region=europe-west1 \
    --metadata 'PIP_PACKAGES=pandas scikit-learn joblib xgboost==1.4.2 gcsfs' \
    --initialization-actions gs://goog-dataproc-initialization-actions-europe-west1/python/pip-install.sh                                                  \
    --master-machine-type=n1-standard-32 \
    --master-boot-disk-size=100

# Experiment - Cluster initialization by providing conda environment
# Error: Component miniconda3 failed to activate
gcloud dataproc clusters create il-l2g-conda \
    --image-version=2.0 \
    --region=europe-west1 \
    --properties='dataproc:conda.env.config.uri=gs://ot-team/irene/l2g/environment.yaml' \
    --master-machine-type=n1-standard-32 \
    --master-boot-disk-size=100
```

### Submit job

```bash
gcloud dataproc jobs submit pyspark 
    --cluster=il-l2g 
    production/predict.py -- 
    --feature_matrix gs://genetics-portal-dev-staging/l2g/220212/gold_standards/featurematrix_w_goldstandards.full.220212.parquet 
    --classifier gs://genetics-portal-dev-staging/l2g/220128/models/xgboost-full_model-high_medium-0.model.joblib.gz 
    --output_file gs://ot-team/irene/l2g/predictions-2022-03-09.parquet
```
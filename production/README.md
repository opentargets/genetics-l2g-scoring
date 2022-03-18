### Start Dataproc cluster

```bash
# Set parameters.
export CLUSTER_NAME=il-l2g
export CLUSTER_REGION=europe-west1

# Cluster initialization with pip
gcloud dataproc clusters create ${CLUSTER_NAME} \
    --image-version=2.0 \
    --region=${CLUSTER_REGION} \
    --metadata 'PIP_PACKAGES=pandas scikit-learn joblib xgboost==1.4.2 gcsfs hydra-core' \
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
# Go to source directory
cd production/src
```

Arguments are passed to the job from the provided configuration file.
```bash
gcloud dataproc jobs submit pyspark predict.py \
    --cluster=${CLUSTER_NAME} \
    --files=conf/config.yaml,utils.py 
```

For testing purposes, it is also possible to override any of the arguments as command line arguments. For example, we can change the location of the model.:
```bash
gcloud dataproc jobs submit pyspark predict.py -- l2g.model=gs://ot-team/irene/l2g/xgboost-full_model-high_medium-1.model.joblib.gz\
    --cluster=${CLUSTER_NAME} predict.py \
    --files=conf/config.yaml,utils.py 
```
> **Note:** It is deeply encouraged to use the configuration file to pass arguments to the job.
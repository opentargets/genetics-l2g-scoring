from typing import Tuple

from pathlib import Path
import gcsfs
import joblib
from psutil import virtual_memory
from pyspark.conf import SparkConf
from pyspark.sql import DataFrame, SparkSession


def detect_spark_memory_limit():
    """Spark does not automatically use all available memory on a machine. When working on large datasets, this may
    cause Java heap space errors, even though there is plenty of RAM available. To fix this, we detect the total amount
    of physical memory and allow Spark to use (almost) all of it."""
    mem_gib = virtual_memory().total >> 30
    return int(mem_gib * 0.9)


def initialize_sparksession() -> SparkSession:
    """Initialize spark session."""

    spark_mem_limit = detect_spark_memory_limit()
    spark_conf = (
        SparkConf()
        .set('spark.driver.memory', f'{spark_mem_limit}g')
        .set('spark.executor.memory', f'{spark_mem_limit}g')
        .set('spark.driver.maxResultSize', '0')
        .set('spark.debug.maxToStringFields', '2000')
        .set('spark.sql.execution.arrow.maxRecordsPerBatch', '500000')
        .set('spark.ui.showConsoleProgress', 'false')
    )
    spark = (
        SparkSession.builder.config(conf=spark_conf)
        .master('local[*]')
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    return spark


def load_model(bucket_name, file_name):
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'{bucket_name}/{file_name}') as f:
        return joblib.load(f)


def parse_gs_url(gs_url: str) -> Tuple[str, str]:
    gs_url_split = gs_url.replace("gs://", "").split("/")
    return '/'.join(gs_url_split[:-1]), gs_url_split[-1]


def write_parquet(data: DataFrame, output_path: str) -> None:
    data.write.format('parquet').mode('overwrite').save(output_path)
    return None


def get_cwd() -> str:
    return Path.cwd()

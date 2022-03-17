from typing import Tuple

import gcsfs
import joblib

def load_model(bucket_name, file_name):
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'{bucket_name}/{file_name}') as f:
        return joblib.load(f)

def parse_gs_url(gs_url: str) -> Tuple[str, str]:
    gs_url_split = gs_url.replace("gs://", "").split("/")
    return '/'.join(gs_url_split[:-1]), gs_url_split[-1]

def write_parquet(data: pd.DataFrame, output_file: str) -> None:
    data.to_parquet(output_file, compression='gzip')
    return None
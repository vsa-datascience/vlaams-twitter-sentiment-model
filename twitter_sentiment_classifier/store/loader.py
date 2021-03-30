"""Load in models from public S3 bucket."""
import os
import re
import shutil
from glob import glob
from pathlib import Path
from typing import List
from urllib.error import HTTPError
from urllib.request import urlretrieve

from tqdm import tqdm

# TODO: Update
S3_URL = "https://production-sentiment-flanders-webapp.s3-eu-west-1.amazonaws.com/sentiment_classifier/"


class NoDataException(Exception):
    """Custom exception thrown when trying to load in data."""
    
    def __init__(self, data_names: List[str], path: Path = Path(__file__).parent) -> None:
        self.message = "Unable to load required data, please download the following:\n"
        for name in data_names:
            self.message += f"\t- {name}\n"
        self.message += f"and store them under {path}"
        super().__init__(self.message)


def tqdm_hook(t):
    """Progressbar functionality during model fetch from S3."""
    last_b = [0]
    
    def update_to(b=1, bsize=1, t_size=None):
        if t_size is not None: t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    
    return update_to


def fetch_data_file(file: str) -> bool:
    """Fetch the requested data file from store."""
    filename = f"{os.path.dirname(__file__)}/data/{file}"
    return os.path.exists(filename)


def fetch_data_file_s3(file: str) -> bool:
    """Fetch the requested data file from S3."""
    filename = f"{os.path.dirname(__file__)}/data/{file}"
    if os.path.exists(filename): return True
    if not os.path.exists(f"{os.path.dirname(__file__)}/data/"): os.makedirs(f"{os.path.dirname(__file__)}/data/")
    
    # tweets_processed_df does not exists locally, check if on server
    try:
        url = f'{S3_URL}{file}'
        with tqdm(
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=f"Downloading '{file}'") as pbar:
            urlretrieve(
                    url,
                    filename=filename,
                    reporthook=tqdm_hook(pbar),
            )
        return True
    except HTTPError:
        raise Exception(f"Unable to load '{file}'!")


def fetch_all_tweet_data():
    """Check if all the tweet-data is stored locally."""
    data_names = [
        'tweets_annotated.jsonl',
        'tweets_sorted.jsonl',
        'tweets_test.jsonl',
        'tweets_train.jsonl',
        'tweets_val.jsonl',
    ]
    
    # Check if all data samples stored locally
    for name in data_names:
        if not fetch_data_file(name):
            raise NoDataException(
                    data_names=data_names,
                    path=Path(__file__).parent / 'data',
            )


def fetch_tweet_dump(output_dir: str):
    """Load all raw twitter-API query-results in zipped format from local store."""
    if os.path.exists(f"{output_dir}/raw_dump"):
        return True
    
    # raw_dump does not exists locally, throw error
    raise NoDataException([
        'raw_dump.zip',
    ])


def fetch_model(
        model: str,
        unzip: bool = False,
) -> None:
    """Fetch model from public S3 bucket, unzip if requested."""
    url = S3_URL + model
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {model}") as pbar:
        urlretrieve(
                url,
                filename=f"{os.path.dirname(__file__)}/{model}",
                reporthook=tqdm_hook(pbar),
        )
    
    # Unzip the fetched object if desired
    if unzip:
        print(f"Unzipping {model}...")
        shutil.unpack_archive(
                f"{os.path.dirname(__file__)}/{model}",
                f"{os.path.dirname(__file__)}/",
        )


def custom_make_archive(source, destination):
    """Custom implementation of the shutil make_archive function."""
    base = os.path.basename(destination)
    name = base.split('.')[0]
    zip_format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, zip_format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, zip_format), destination)


def zip_model(
        model: str,
) -> None:
    """Zip the requested model."""
    custom_make_archive(  # Create zips using shutil.make_archive()
            source=f"{os.path.dirname(__file__)}/{model}",
            destination=f"{os.path.dirname(__file__)}/{model}.zip",
    )


def load_zipped_model(name: str, overwrite: bool = False) -> bool:
    """Load in the language model, return True if exists (downloaded or already on machine)."""
    # Check if model exists locally
    if os.path.exists(f"{os.path.dirname(__file__)}/{name}/pytorch_model.bin") and not overwrite:
        return True
    
    # Model does not exists locally, check if on server
    try:
        model = re.sub(r'-v[0-9]+$', '', name)
        fetch_model(model=f'{model}.zip', unzip=True)
        return True
    except HTTPError:
        # No pre-trained model available
        print(f"No pre-trained model for '{name}' available!")
        return False


def clean_store():
    """Delete all non-Python files present in the store."""
    non_python = set(glob(f'{os.path.dirname(__file__)}/*')) - set(glob(f'{os.path.dirname(__file__)}/*.py'))
    for path in non_python:
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

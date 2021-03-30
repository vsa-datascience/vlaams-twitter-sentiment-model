"""Load in models from public S3 bucket."""
import os
import re
import shutil
from glob import glob
from urllib.error import HTTPError
from urllib.request import urlretrieve

from tqdm import tqdm

S3_URL = "https://production-sentiment-flanders-webapp.s3-eu-west-1.amazonaws.com/sentiment_classifier/"


def tqdm_hook(t):
    """Progressbar functionality during model fetch from S3."""
    last_b = [0]
    
    def update_to(b=1, bsize=1, t_size=None):
        if t_size is not None: t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    
    return update_to


def fetch_data_file(file: str):
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
    """Fetch all the tweet-data from S3."""
    fetch_data_file('tweets_annotated.jsonl')
    fetch_data_file('tweets_sorted.jsonl')
    fetch_data_file('tweets_test.jsonl')
    fetch_data_file('tweets_train.jsonl')
    fetch_data_file('tweets_val.jsonl')


def fetch_tweet_dump(output_dir: str):
    """Load all raw twitter-API query-results in zipped format from s3."""
    filename = f"{os.path.dirname(__file__)}/raw_dump.zip"
    if os.path.exists(f"{output_dir}/raw_dump"): return True
    
    # raw_dump does not exists locally, check if on server
    try:
        url = f'{S3_URL}raw_dump.zip'
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading raw_dump") as pbar:
            urlretrieve(
                    url,
                    filename=filename,
                    reporthook=tqdm_hook(pbar),
            )
        print(f"Unzipping raw_dump...")
        shutil.unpack_archive(
                f"{os.path.dirname(__file__)}/raw_dump.zip",
                f"{output_dir}/raw_dump",
        )
        return True
    except HTTPError:
        raise Exception("Unable to load test-dataset!")


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

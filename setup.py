import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    # Necessary requirements
    'scikit-learn',
    'torch',
    'transformers',
    'tqdm',  # Do not specify version, may conflict with other packages
    
    # Development requirements, see requirements_dev.txt
    # 'emoji~=0.6.0',
    # 'pandas~=1.1.4',
    # 'pandas_profiling',
    # 'tensorflow',
    # 'tensorflow_hub',
    # 'tensorflow_text',
    # 'umap-learn',
    # 'lit-nlp',
    # 'tensorflow_datasets',
]

# noinspection SpellCheckingInspection
setup(
        name="twitter-sentiment-classifier",
        version="0.1.1",
        description="Define the sentiment of Dutch tweets by means of a Transformer model.",
        long_description=README,
        long_description_content_type="text/markdown",
        url="https://gitlab.com/radix-ai/statistiek-vlaanderen/twitter-sentiment-classifier",
        author="radix.ai",
        author_email="developers@radix.ai",
        license="CC BY-NC-ND 3.0",
        classifiers=["Programming Language :: Python :: 3", "Programming Language :: Python :: 3.8", ],
        packages=find_packages(exclude=("tests", "notebooks", "doc", "scripts")),
        include_package_data=True,
        package_data={"": ["data/synonym_config.pkl"]},
        install_requires=INSTALL_REQUIRES,
)

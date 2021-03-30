# TwitterSentimentClassifier

Deep Learning model behind the Vlaams Twitter Sentiment webapplication, developed by [Radix](http://radix.ai).

## 1. Installation

```text
pip install git+https://github.com/vsa-datascience/vlaams-twitter-sentiment-model.git
```


## 2. Usage

```python
from twitter_sentiment_classifier import batch_predict

texts = [
    'Ik haat u!',  # Negative
    'Daar ben ik het mee eens',  # Neutral
    'Ik hou van je!'  # Positive
]

batch_predict(texts)
```


## 3. Development

In order to make changes to the project, run the Language Interpretability Tool, or run the notebooks, the developer
requirements must be installed. These can (only manually) be installed by running:

```
pip install -r requirements_dev.txt
```


## 4. Main Classes

### 4.1. SentimentModel

The core class within the project is the `SentimentModel`, found in `sentiment_model.py`. This model is a wrapper for
the HuggingFace Transformer model (`RobBERT`) and handles tasks as:

- Loading and saving the model
- Download pre-trained model from the `S3 bucket` if not yet available
- Provide a script for training and evaluation
- Reset functionality
- Logging

### 4.2. Encoder

The `Encoder` class extracts the encoder-part from the `SentimentModel` and creates a wrapper for this. This class is
only used to encode sentences using the trained language model.

### 4.3. Classifier

The `Classifier` class extracts the classification-head from the `SentimentModel` and creates a wrapper for this. This
class takes as inputs the encoded sentences (see `Encoder`) and is mainly used to train the classification head. If one
would only want to fine-tune the head, it is recommended using this via the `Classifier` instead of using the
`SentimentModel` for performance reasons (~1min runtime compared to several hours for every epoch).


## 5. Language Interpretability Tool

To run the Language Interpretability Tool (LIT), execute the `lit_tool.py` script.


## 6. Notebooks

Every step in the model's timeline is explained through a guiding notebook, that can be found in the `notebooks/`
folder. Aside from `0. Data Parsing`, all the data is provided to run the corresponding notebook.


## 7. Store

Previously created data related to the project is stored in an online `S3 bucket`. In order to download this data, a
`store` with a corresponding `loader.py` script is created. The methods found under `loader.py` are used to download
specific elements from `S3`, all the downloaded data is stored in the `store/` folder. If one wants to remove all
downloaded data, run:

```python
from twitter_sentiment_classifier import clean_store

clean_store()
```

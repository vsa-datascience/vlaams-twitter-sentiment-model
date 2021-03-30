"""Complete model, which combines both the encoder (see encoder.py) and classifier (see classifier.py) part."""
import logging
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer, Trainer, \
    TrainingArguments

from twitter_sentiment_classifier.dataset import ClassificationDataset, CompleteDataset, split_sample
from twitter_sentiment_classifier.store.loader import load_zipped_model

SENTIMENT2LABEL = {
    'NEGATIVE': 0,
    'NEUTRAL':  1,
    'POSITIVE': 2,
}
LABEL2SENTIMENT = {
    v: k for k, v in SENTIMENT2LABEL.items()
}


class SentimentModel:
    def __init__(
            self,
            batch_size: int = 16,
            cuda: bool = torch.cuda.is_available(),
            n_classes: int = 3,
            version: int = 0,
    ):
        """
        Complete sentiment model which takes as inputs raw sentences and outputs sentiment predictions.
        
        :param batch_size: Number of samples encoded in single batch
        :param cuda: Use CUDA-cores (GPU) during encoding, which leads to significant performance improvements
        :param n_classes: Number of output-classes
        :param version: Model version
        """
        self.n_classes: int = n_classes
        self.version: int = version
        self.batch_size: int = batch_size
        self._use_cuda: bool = cuda
        
        # Model's core parts
        self._model: Optional[RobertaForSequenceClassification] = None
        self._tokenizer: Optional[RobertaTokenizer] = None
        
        # Default parameters
        self.path_store: str = f"{os.path.dirname(__file__)}/store/{self}/"
        self.path_to_encodings = Path(os.path.join(self.path_store, 'encodings'))
        self.logger = None
        self.__init_dirs()
        self.__load()
    
    def __str__(self):
        """Shared model representation."""
        return f"sentiment-model-classes{self.n_classes}{f'-v{self.version}' if self.version else ''}"
    
    def __repr__(self):
        """String-representation, unique for both the Classifier and Encoder components."""
        return (
            f"SentimentModel{f'V{self.version}' if self.version else ''}(\n"
            f"\tn_classes={self.n_classes}\n"
            f"\tbatch_size={self.batch_size}\n"
            f"\tuse_cuda={self._use_cuda}\n"
            f")"
        )
    
    def __call__(self, texts: List[str]):
        """Query the model to create an encoding."""
        with torch.no_grad():
            tokenized = self.tokenize(texts)
            if self._use_cuda: tokenized.to('cuda')
            return self._model(**tokenized)[0].to("cpu").detach().argmax(axis=1)
    
    def create_dataset(self, samples: List[Dict[str, Any]]) -> CompleteDataset:
        """Create a dataset that fits the model for the given samples."""
        _, texts, labels = zip(*[split_sample(sample) for sample in samples])
        tokens = self.tokenize(texts)
        return CompleteDataset(tokens=tokens, labels=labels)
    
    def get_model(
            self,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
    ) -> RobertaForSequenceClassification:
        """Return the RobBERT model stored in the SentimentModel, possible with debugging functionality."""
        # If no debugging functionality necessary, return default model
        if not output_hidden_states and not output_attentions:
            return self._model
        
        # Debugging version of the model, create new model to return
        model_config = RobertaConfig.from_pretrained(
                self.path_store,
                num_labels=self.n_classes,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
        )
        model = RobertaForSequenceClassification.from_pretrained(
                self.path_store,
                config=model_config,
        )
        if self._use_cuda: model.to("cuda")
        model.eval()  # Put model in inference-mode by default
        return model
    
    def get_tokenizer(self):
        """Return the RobBERT tokenizer stored in the SentimentModel."""
        return self._tokenizer
    
    def tokenize(self, texts):
        """Tokenize the given texts using the RobBERT tokenizer."""
        return self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',  # Return PyTorch array
        )
    
    def reset(self, initialise: bool = True) -> None:
        """
        Reset the model.
        
        :param initialise: Initialise the Classifier's head with that found on server
        """
        # Load pre-trained model from S3
        if initialise:
            self.logger.warning("Re-initialising RobBERT and initialising...")
            load_zipped_model(name=str(self), overwrite=True)
            self.logger.warning("--> Successful")
        
        # Load RobBERT model from HuggingFace
        else:
            self.logger.warning("Re-initialising RobBERT without initialisation...")
            self._model = RobertaForSequenceClassification.from_pretrained(
                    "pdelobelle/robbert-v2-dutch-base",
                    num_labels=self.n_classes,
            )
            self._model.save_pretrained(self.path_store)  # Overwrite current model in store
            self.logger.warning("--> Successful")
    
    def train(
            self,
            n_epochs: int,
            train_samples: List[Dict[str, Any]],
            val_samples: List[Dict[str, Any]],
            autosave: bool = True,
            n_steps_logging: int = 200,  # Costly, so don't do it too frequently
            n_last_layers: int = 1,
            test_samples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Train the sentiment model.

        Select only a few of the model's last layers that are trained, this makes that most of the pre-trained knowledge
        of the language model remains, while still giving the freedom to connect the language model with the
        classification head meaningfully.

        :note: The more layers you want to train, the more memory is required

        :param n_epochs: Number of training epochs
        :param train_samples: Training samples
        :param val_samples: Validation samples
        :param autosave: Automatically save the trained model
        :param n_steps_logging: Number of training steps between logging
        :param n_last_layers: Number of last layers of the model that are going to be trained
        :param test_samples: Testing samples, used to return test statistics of the newly trained model
        :return: Test statistics if test_dataset is given, None otherwise
        """
        # Create the datasets
        train_data = self.create_dataset(train_samples)
        val_data = self.create_dataset(val_samples)
        
        # Put the model in training-mode
        self._model.train()
        
        # Freeze the complete model
        for param in self._model.base_model.parameters():
            param.requires_grad = False
        
        # Un-freeze the last n layers
        for n in range(1, n_last_layers + 1):
            for param in self._model.roberta.encoder.layer[-n].parameters():
                param.requires_grad = True
        
        # Setup training environment
        trainer_args = TrainingArguments(
                output_dir=os.path.join(self.path_store, '.checkpoints'),  # output directory
                num_train_epochs=n_epochs,  # total # of training epochs
                per_device_train_batch_size=self.batch_size,  # batch size per device during training
                per_device_eval_batch_size=self.batch_size,  # batch size for evaluation
                weight_decay=0.01,  # strength of weight decay
                logging_dir=os.path.join(self.path_store, '.log-complete'),  # directory for storing logs
                evaluation_strategy="steps",  # perform validation evaluations during training
                logging_steps=n_steps_logging,  # Number of steps between validation
        )
        trainer = Trainer(
                model=self._model,  # the model to be trained
                args=trainer_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # compute various metrics during training
                train_dataset=train_data,  # training dataset
                eval_dataset=val_data,  # validation dataset
        )
        
        # Trigger the trainer
        trainer.train()
        
        # Put model back to eval() to disable dropout
        self._model.eval()
        
        # Remove all previous encodings (if already created) since the encoder changed
        for f in glob(os.path.join(self.path_to_encodings, '*')):
            os.remove(f)
        
        # Save the model
        if autosave: self.__save()
        
        # Evaluate the final model if test data is provided
        if test_samples is not None:
            test_data = self.create_dataset(test_samples)
            return trainer.predict(test_dataset=test_data)
    
    def eval(
            self,
            test_samples: List[Dict[str, Any]],
    ):
        """
        Evaluate the model on the provided test-samples.
        
        :param test_samples: Testing samples, used to return test statistics of the newly trained model
        :return: Test statistics
        """
        # Put test data in dataset
        test_data = self.create_dataset(test_samples)
        
        # Ensure that model is in evaluation mode
        self._model.eval()
        
        # Setup environment to perform testing in
        trainer_args = TrainingArguments(
                output_dir=os.path.join(self.path_store, '.checkpoints'),  # output directory
                per_device_train_batch_size=self.batch_size,  # batch size per device during training
                per_device_eval_batch_size=self.batch_size,  # batch size for evaluation
        )
        trainer = Trainer(
                model=self._model,  # the model to be trained
                args=trainer_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # compute various metrics during training
        )
        
        # Perform evaluation and return results
        return trainer.predict(test_dataset=test_data)
    
    def __init_logger(self):
        """Initialise the logger, can only be done once model directory exists."""
        logging.basicConfig(
                filename=os.path.join(self.path_store, 'logging.log'),
                format="%(asctime)s %(message)s",
                filemode="a",
        )
        self.logger = logging.getLogger(name="my_logger")
        self.logger.setLevel(logging.DEBUG)
    
    def __init_dirs(self) -> None:
        """Initialise dictionary to store the models in, load pre-trained models from S3 buckets if exist."""
        if not os.path.exists(self.path_to_encodings): os.makedirs(self.path_to_encodings)
        if load_zipped_model(name=str(self)):
            # Pre-trained folder already exists, load in logger
            self.__init_logger()
        else:
            # No pre-trained model found, initialise new
            os.mkdir(self.path_store)
            self.__init_logger()
            self.logger.warning("No pre-trained model found, initialising one from scratch")
            self.reset(initialise=False)
            print("No pre-trained model found, initialised new model. Classification head not yet trained.")
    
    def __load(self):
        """Load in the model specified by the path. Can be both a Huggingface Transformer or locally stored."""
        self.logger.info("Loading SentimentModel...")
        self._tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')
        if load_zipped_model(name=str(self)):
            self._model = RobertaForSequenceClassification.from_pretrained(self.path_store, num_labels=self.n_classes)
            if self._use_cuda: self._model.to("cuda")
            self._model.eval()  # Put model in inference-mode by default
        else:
            raise ModuleNotFoundError(f"No model found for '{self}'!")
        self.logger.info("--> Successful")
    
    def __save(self):
        """Save the complete RobBERT model."""
        self.logger.info(f"Saving model '{self}'...")
        self._model.save_pretrained(self.path_store)
        self.logger.info("--> Successful")


class Encoder(SentimentModel):
    
    def __init__(
            self,
            batch_size: int = 64,
            cuda: bool = torch.cuda.is_available(),
            n_classes: int = 3,
            version: int = 0,
    ):
        """
        Encoder part of the sentiment model which takes as inputs raw sentences and outputs sentiment embeddings.
        
        :param batch_size: Number of samples encoded in single batch
        :param cuda: Use CUDA-cores (GPU) during encoding, which leads to significant performance improvements
        :param n_classes: Number of output-classes
        :param version: Model version
        """
        super(Encoder, self).__init__(
                batch_size=batch_size,
                cuda=cuda,
                n_classes=n_classes,
                version=version,
        )
        
        # Extract the encoder from the model
        self._encoder = RobertaModel.from_pretrained(self.path_store)
        if self._use_cuda: self._encoder.to("cuda")
    
    def __repr__(self):
        """String-representation, unique for both the Classifier and Encoder components."""
        return (
            f"Encoder{f'V{self.version}' if self.version else ''}(\n"
            f"\tn_classes={self.n_classes}\n"
            f"\tbatch_size={self.batch_size}\n"
            f"\tuse_cuda={self._use_cuda}\n"
            f")"
        )
    
    def __call__(self, texts: List[str]) -> None:
        """Query the model to create an encoding."""
        with torch.no_grad():
            tokenized = self.tokenize(texts)
            if self._use_cuda: tokenized.to('cuda')
            return self._encoder(**tokenized)[0][:, 0, :].to("cpu").detach()
    
    def get_encoding(self, idx: int):
        """Get the encoding for the given index."""
        return np.load(self.path_to_encodings / (str(idx) + '.npz'))['arr_0']
    
    def encode_data(
            self,
            data_samples,
    ) -> None:
        """Encode all the tweet samples."""
        # Split the samples
        data_ids, data_texts, _ = zip(*[split_sample(sample) for sample in data_samples])
        
        # Check which encodings are already created
        already_created = {x.stem for x in self.path_to_encodings.glob("*.npz")}
        
        # Collect all the texts that are not yet encoded
        encoding_ids, encoding_texts = [], []
        for idx, text in zip(data_ids, data_texts):
            if str(idx) in already_created: continue
            encoding_ids.append(idx)
            encoding_texts.append(text)
            already_created.add(idx)
        self.logger.info(f"Encoding {len(encoding_ids)} tweets")
        
        # Loop over the data by batches to encode
        for i in tqdm(range(0, len(encoding_ids), self.batch_size)):
            encodings = self(texts=encoding_texts[i:i + self.batch_size]).numpy()
            
            # Merge the encodings with the corresponding tweet
            for enc_idx, tweet_idx in enumerate(encoding_ids[i:i + self.batch_size]):
                np.savez_compressed(
                        self.path_to_encodings / str(tweet_idx),
                        encodings[enc_idx],
                        allow_pickle=True,
                )
        
        # Clear CUDA cache when finished
        if self._use_cuda: torch.cuda.empty_cache()


class Classifier(SentimentModel):
    
    def __init__(
            self,
            batch_size: int = 1024,
            cuda: bool = torch.cuda.is_available(),
            n_classes: int = 3,
            version: int = 0,
    ):
        """
        Classifier part of the sentiment model which takes as inputs the encoded sentences and outputs sentiment labels.
        
        :param batch_size: Number of samples encoded in single batch
        :param cuda: Use CUDA-cores (GPU) during encoding, which leads to significant performance improvements
        :param n_classes: Number of output-classes
        :param version: Model version
        """
        super(Classifier, self).__init__(
                batch_size=batch_size,
                cuda=cuda,
                n_classes=n_classes,
                version=version,
        )
        
        # Extract the encoder from the model
        self._classifier = RobertaClassificationHead(config=self._model.config)
        self._classifier.load_state_dict(self._model.classifier.state_dict())
        if self._use_cuda: self._classifier.to("cuda")
    
    def __repr__(self):
        """String-representation, unique for both the Classifier and Encoder components."""
        return (
            f"Classifier{f'V{self.version}' if self.version else ''}(\n"
            f"\tn_classes={self.n_classes}\n"
            f"\tbatch_size={self.batch_size}\n"
            f"\tuse_cuda={self._use_cuda}\n"
            f")"
        )
    
    def __call__(self, embeddings: List[Any]):
        """Query the model to create an encoding."""
        if type(embeddings) == list or type(embeddings) == tuple:
            embeddings = torch.tensor(embeddings)
        with torch.no_grad():
            prediction = self._classifier(embeddings, )
            return prediction.argmax(axis=1), softmax(prediction, dim=-1)
    
    def create_dataset(self, samples: List[Dict[str, Any]]) -> ClassificationDataset:
        """Create a dataset that fits the model for the given samples."""
        # Check if all samples are encoded first
        already_created = {x.stem for x in self.path_to_encodings.glob("*.npz")}
        for s in samples:
            assert str(s['id']) in already_created
        
        # Fetch encodings and put these in the dataset
        ids, _, labels = zip(*[split_sample(sample) for sample in samples])
        embeddings = [np.load(self.path_to_encodings / (str(idx) + '.npz'))['arr_0'] for idx in ids]
        return ClassificationDataset(embeddings=embeddings, labels=labels)
    
    def train(
            self,
            n_epochs: int,
            train_samples: List[Dict[str, Any]],
            val_samples: List[Dict[str, Any]],
            autosave: bool = True,
            n_steps_logging: int = 100,
            test_samples: Optional[List[Dict[str, Any]]] = None,
            **kwargs,  # Ignored
    ):
        """
        Train the head of the sentiment model.

        Select only a few of the model's last layers that are trained, this makes that most of the pre-trained knowledge
        of the language model remains, while still giving the freedom to connect the language model with the
        classification head meaningfully.

        :note: The more layers you want to train, the more memory is required

        :param n_epochs: Number of training epochs
        :param train_samples: Training samples
        :param val_samples: Validation samples
        :param autosave: Automatically save the trained model
        :param n_steps_logging: Number of training steps between logging
        :param test_samples: Testing samples, used to return test statistics of the newly trained model
        :return: Test statistics if test_dataset is given, None otherwise
        """
        # Create the datasets
        train_data = self.create_dataset(train_samples)
        val_data = self.create_dataset(val_samples)
        
        # Put the model in training-mode
        self._classifier.train()
        
        # Setup training environment
        trainer_args = TrainingArguments(
                output_dir=os.path.join(self.path_store, '.checkpoints'),  # output directory
                num_train_epochs=n_epochs,  # total # of training epochs
                per_device_train_batch_size=self.batch_size,  # batch size per device during training
                per_device_eval_batch_size=self.batch_size,  # batch size for evaluation
                weight_decay=0.01,  # strength of weight decay
                logging_dir=os.path.join(self.path_store, '.log-fine-tune'),  # directory for storing logs
                evaluation_strategy="steps",  # perform validation evaluations during training
                logging_steps=n_steps_logging,  # Number of steps between validation
        )
        trainer = Trainer(
                model=self._classifier,  # the model to be trained
                args=trainer_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # compute various metrics during training
                train_dataset=train_data,  # training dataset
                eval_dataset=val_data,  # validation dataset
        )
        
        # Trigger the trainer
        trainer.train()
        
        # Put model back to eval() to disable dropout
        self._classifier.eval()
        
        # Save the model
        if autosave: self.__save()
        
        # Evaluate the final model if test data is provided
        if test_samples is not None:
            test_data = self.create_dataset(test_samples)
            return trainer.predict(test_dataset=test_data)
    
    def eval(
            self,
            test_samples: List[Dict[str, Any]],
    ):
        """
        Evaluate the model on the provided test-samples.
        
        :param test_samples: Testing samples, used to return test statistics of the newly trained model
        :return: Test statistics
        """
        # Put test data in dataset
        test_data = self.create_dataset(test_samples)
        
        # Ensure that model is in evaluation mode
        self._classifier.eval()
        
        # Setup environment to perform testing in
        trainer_args = TrainingArguments(
                output_dir=os.path.join(self.path_store, '.checkpoints'),  # output directory
                per_device_train_batch_size=self.batch_size,  # batch size per device during training
                per_device_eval_batch_size=self.batch_size,  # batch size for evaluation
        )
        trainer = Trainer(
                model=self._classifier,  # the model to be trained
                args=trainer_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # compute various metrics during training
        )
        
        # Perform evaluation and return results
        return trainer.predict(test_dataset=test_data)
    
    def __save(self):
        """Save the classifier by modifying the general model's head."""
        self.logger.info(f"Saving classification head of model '{self}'...")
        self._model.classifier.load_state_dict(self._classifier.state_dict())
        self._model.save_pretrained(self.path_store)
        self.logger.info("--> Successful")


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    
    def __init__(self, config):
        """Initialise the head as a copy of RobertaForSequenceClassification."""
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
    
    def forward(self, features, labels=None):
        """Query the model."""
        assert len(features.shape) == 2  # Batch over the <s> encoded tokens (equiv. to [CLS])
        x = features[:, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        
        # From RobertaForSequenceClassification
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)
        else:
            outputs = logits
        
        return outputs  # (loss), logits


def compute_metrics(pred):
    """Compute relevant metrics during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def batch_predict(
        texts: List[str],
        model: Optional[SentimentModel] = None,
        batch_size: int = 32,
) -> List[str]:
    """
    Create predictions by batches.

    :param texts: List of texts for which a prediction must be made
    :param model: SentimentModel, a new one is created if not provided
    :param batch_size: Inference batch-size
    :return: List of predictions
    """
    # Load new pipeline if none is provided
    if model is None: model = SentimentModel()
    
    # Create the predictions
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting..."):
        predictions += model(texts[i:i + batch_size]).tolist()
    assert len(predictions) == len(texts)
    
    # Transform LABEL_X predictions to readable ones.
    for i, p in enumerate(predictions):
        predictions[i] = LABEL2SENTIMENT[p]
    return predictions

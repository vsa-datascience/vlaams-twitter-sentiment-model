"""Datasets used to feed the models during training."""
from typing import Any, Dict, Optional, Tuple

import torch

LABEL2IDX: Dict[str, int] = {
    'NEGATIVE': 0,
    'NEUTRAL':  1,
    'POSITIVE': 2,
}


class ClassificationDataset(torch.utils.data.Dataset):
    
    def __init__(self, embeddings: list, labels: list):
        """
        Dataset used to fine-tune only the classification head.
        
        :note: This dataset is only conform with Classifier.
        
        :param embeddings: Sentence embeddings as a result of the encoder
        :param labels: Target labels as a list of indices
        """
        self.embeddings = embeddings
        self.labels = labels
    
    def __str__(self):
        return f"ClassificationDataset(n_samples={len(self)})"
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, idx):
        """Get a data sample in the format that suites the Classifier model."""
        item = {
            "features": torch.tensor(self.embeddings[idx]),
            "labels":   torch.tensor(self.labels[idx]),
        }
        return item
    
    def __len__(self):
        """Length of the dataset."""
        return len(self.labels)


class CompleteDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokens, labels):
        """
        Dataset used to train the complete model.
        
        :note: This dataset is only conform with SentimentModel.
        
        :param tokens: Tokenized input texts
        :param labels: Target labels as a list of indices
        """
        self.tokens = tokens
        self.labels = labels
    
    def __str__(self):
        return f"CompleteDataset(n_samples={len(self)})"
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, idx):
        """Get a data sample in the format that suites the Encoder model."""
        item = {key: val[idx] for key, val in self.tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """Length of the dataset."""
        return len(self.labels)


def split_sample(sample: Dict[str, Any]) -> Tuple[str, str, Optional[int]]:
    """Split the sample in id, text, and label."""
    label = get_label_idx(sample['label']) if 'label' in sample.keys() else None
    return sample['id'], sample['text'], label


def get_label_idx(label: str) -> int:
    """Get the corresponding label index."""
    return LABEL2IDX[label]

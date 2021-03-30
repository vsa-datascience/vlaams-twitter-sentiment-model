"""
Get insights in the model using the Language Interpretability Tool.

Sources:
 - General: https://pair-code.github.io/lit/setup/
 - WrappedModel: https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/sst_pytorch_demo.py
 - WrappedDataset: https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/datasets/glue.py
"""
import json
import os
import re
from random import sample
from typing import Optional

import pandas as pd
import torch
from absl import app
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model, types as lit_types
from lit_nlp.api.dataset import Dataset
from lit_nlp.dev_server import Server
from lit_nlp.lib import utils
from torch.nn.functional import softmax

from sentiment_model import SentimentModel
from store.loader import fetch_all_tweet_data


class WrappedModel(lit_model.Model):
    """Wrapper for out SentimentModel in order to be used by LIT."""
    LABELS = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    compute_grads = True  # Enable gradient to get more functionality
    
    def __init__(self):
        # Load in the debugging-version of the sentiment model
        self.sentiment_model = SentimentModel()
        
        # Use the RobBERT model and tokenizer
        self.model = self.sentiment_model.get_model(
                output_hidden_states=True,
                output_attentions=True,
        )
        self.tokenizer = self.sentiment_model.get_tokenizer()
    
    def max_minibatch_size(self):
        """Tell the lit_model.Model.predict() how many inputs per batch."""
        return 16
    
    def predict_minibatch(self, inputs):
        """Make predictions for the given batch of inputs."""
        # Preprocess to ids and masks, and make the input batch.
        encoded_input = self.sentiment_model.tokenize([inp["tweet"] for inp in inputs])
        
        # Check and send to cuda (GPU) if available
        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                encoded_input[tensor] = encoded_input[tensor].cuda()
        
        # Run a forward pass.
        with torch.set_grad_enabled(self.compute_grads):
            logits, embs, unused_attentions = self.model(**encoded_input).values()
        
        # Post-process outputs.
        batched_outputs = {
            "probas":    softmax(logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "ntok":      torch.sum(encoded_input["attention_mask"], dim=1),
            "cls_emb":   embs[-1][:, 0],  # last layer, first token (is the cls token that's used for classification)
        }
        
        # Add attention layers to batched_outputs
        for i, layer_attention in enumerate(unused_attentions):
            batched_outputs[f"layer_{i}/attention"] = layer_attention
        
        # Request gradients after the forward pass.
        # Note: embs[0] includes position and segment encodings, as well as sub-word embeddings.
        if self.compute_grads:
            # <torch.float32>[batch_size, num_tokens, emb_dim]
            scalar_pred_for_gradients = torch.max(
                    batched_outputs["probas"],
                    dim=1,
                    keepdim=False,
                    out=None,
            )[0]
            batched_outputs["input_emb_grad"] = torch.autograd.grad(
                    scalar_pred_for_gradients,
                    embs[0],
                    grad_outputs=torch.ones_like(scalar_pred_for_gradients),
            )[0]
        
        # Return as NumPy for further processing.
        detached_outputs = {k: v.cpu().detach().numpy() for k, v in batched_outputs.items()}
        
        # Unbatch outputs so we get one record per input example.
        for output in utils.unbatch_preds(detached_outputs):
            ntok = output.pop("ntok")
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(
                    output.pop("input_ids")[:ntok])
            
            # set token gradients
            if self.compute_grads:
                output["token_grad_sentence"] = output["input_emb_grad"][:ntok]
            
            # Process attention.
            for key in output:
                if not re.match(r"layer_(\d+)/attention", key):
                    continue
                # Select only real tokens, since most of this matrix is padding.
                # <float32>[num_heads, max_seq_length, max_seq_length]
                # -> <float32>[num_heads, num_tokens, num_tokens]
                output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
                # Make a copy of this array to avoid memory leaks, since NumPy otherwise
                # keeps a pointer around that prevents the source array from being GCed.
                output[key] = output[key].copy()
            yield output
    
    def input_spec(self) -> lit_types.Spec:
        """Give the input specifications."""
        return {
            "tweet": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS, required=False)
        }
    
    def output_spec(self) -> lit_types.Spec:
        """Give the output specifications."""
        ret = {
            "tokens":  lit_types.Tokens(),
            "probas":  lit_types.MulticlassPreds(parent="label", vocab=self.LABELS),
            "cls_emb": lit_types.Embeddings()
        }
        
        # Gradients, if requested.
        if self.compute_grads:
            ret["token_grad_sentence"] = lit_types.TokenGradients(align="tokens")
        
        # Attention heads, one field for each layer.
        for i in range(self.model.config.num_hidden_layers):
            ret[f"layer_{i}/attention"] = lit_types.AttentionHeads(align=("tokens", "tokens"))
        return ret


class WrappedDataset(Dataset):
    LABELS = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    ANNOTATORS = ['A', 'H', 'I', 'O', 'R', 'S']
    
    def __init__(self, path, n_sample: Optional[int] = None):
        """
        Wrap the annotated twitter data to be readable by LIT.
        
        :param path: Path to the data to use
        :param n_sample: Optional random number of datapoints to be sample, use all datapoints if not provided
        """
        # Load in the test-data
        with open(path, 'r') as f:
            annotations_test = [json.loads(line) for line in f.readlines()]
        
        # Sample if requested
        if n_sample:
            annotations_test = sample(annotations_test, n_sample)
            print(f"Sampled {len(annotations_test)} samples")
        
        # Transform the data to a pandas DataFrame (keep only the relevant columns)
        df = pd.DataFrame(
                annotations_test,
        )
        df.set_index('id', inplace=True)
        df = df[['text', 'label', 'annotator']]
        
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{
            'tweet':     row['text'],
            'label':     row['label'],
            'annotator': row['annotator'],
        } for _, row in df.iterrows()]
    
    def spec(self):
        return {
            'tweet':     lit_types.TextSegment(),
            'label':     lit_types.CategoryLabel(vocab=self.LABELS),
            'annotator': lit_types.CategoryLabel(vocab=self.ANNOTATORS),
        }


def main(_):
    """Run the Language Interpretability Tool."""
    fetch_all_tweet_data()
    
    # Specify the datasets used for analysis
    datasets = {
        'test_dataset': WrappedDataset(
                path=os.path.expanduser('store/data/tweets_test.jsonl'),
                n_sample=100,
        ),
        'val_dataset':  WrappedDataset(
                os.path.expanduser('store/data/tweets_val.jsonl'),
                n_sample=100,
        ),
    }
    
    # Specify the model used for analysis
    models = {
        'sentiment_model': WrappedModel(),
    }
    
    lit_demo = Server(
            models,
            datasets,
            **server_flags.get_flags()
    )
    lit_demo.serve()


if __name__ == '__main__':
    # Run the Language Interpretability Tool
    app.run(main)

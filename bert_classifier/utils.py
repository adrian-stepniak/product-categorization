import os
import torch
from transformers import PreTrainedTokenizerFast, RobertaModel, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch import tensor
from typing import List
import numpy as np


def get_pretrained_model(model_dir="roberta_base_transformers", freeze_model=True):
    roberta_base_model: RobertaModel = AutoModel.from_pretrained(model_dir)
    # freeze all the parameters
    if freeze_model:
        for param in roberta_base_model.parameters():
            param.requires_grad = False

    return roberta_base_model


def tokenize_data(
        data: List,
        labels: List,
        model_dir: str,
        tokenizer_pad_token: str = '<pad>',
        max_seq_len=100,
        batch_size=16) -> DataLoader:
    """
    Tokenize data with labels to BERT format (tokens, masks)

    :param data: list with text for tokenization
    :param labels: list with labels for data
    :param model_dir: path to the pretrained model
    :param tokenizer_pad_token: padding token used by model
    :param max_seq_len: maximum length of the tokenized data sequence. Longer sequences will be truncated
    :param batch_size: number of samples that will be send by DataLoader in each step
    :return: DataLoader
    """

    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = tokenizer_pad_token

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        data,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    train_seq = tensor(tokens_train['input_ids'])
    train_mask = tensor(tokens_train['attention_mask'])
    train_y = tensor(labels).to(dtype=torch.long)

    # wrap tensors
    tensor_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    tensor_data_sampler = RandomSampler(tensor_data)

    # dataLoader for train set
    return DataLoader(tensor_data, sampler=tensor_data_sampler, batch_size=batch_size, )

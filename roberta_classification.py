import os
import urllib.request

# # TODO: check if exists
# urllib.request.urlretrieve(
#     'https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_transformers.zip',
#     'roberta_base_transformers.zip')
#
# from zipfile import ZipFile
#
# with ZipFile('roberta_base_transformers.zip', 'r') as zipObj:
#     # Extract all the contents of zip file in current directory
#     zipObj.extractall(path='roberta_base_transformers')

import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('dataset_preprocessed.csv', encoding='utf8')
del df['Unnamed: 0']

#
# train_text, val_text, train_labels, val_labels = train_test_split(df['desc_long'], df['category_int'],
#                                                                   random_state=3333,
#                                                                   test_size=0.2,
#                                                                   stratify=df['category_int'])
#
# train_dataset = pd.DataFrame(zip(train_text, train_labels))
# val_dataset = pd.DataFrame(zip(val_text, val_labels))
#
# train_dataset.to_csv('train_dataset.csv')
# val_dataset.to_csv('val_dataset.csv')
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')

train_df.columns = ['index', 'text', 'label']
val_df.columns = ['index', 'text', 'label']

train_text = train_df['text']
val_text = train_df['text']
train_labels = train_df['label']
val_labels = train_df['label']


import torch, os
from transformers import RobertaModel, AutoModel, PreTrainedTokenizerFast

model_dir = "roberta_base_transformers"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
roberta_base_model: RobertaModel = AutoModel.from_pretrained(model_dir)

# example = df['desc_long'][0]
# input = tokenizer.encode(example)
# output = roberta_base_model(torch.tensor([input]))[0]
#
# seq_len = [len(i.split()) for i in train_text]

tokenizer.pad_token = '<pad>'
print(roberta_base_model.config.vocab_size)

max_seq_len = 100
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=max_seq_len,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=max_seq_len,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# define a batch size
batch_size = 64

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in roberta_base_model.parameters():
    param.requires_grad = False

import torch.nn as nn


class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 11)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)

        return x


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(roberta_base_model)
device = torch.device("cuda")
# push the model to GPU
model = model.to(device)
from transformers import AdamW
from sklearn.metrics import accuracy_score

optimizer = AdamW(model.parameters(), lr=1e-3)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print(class_wts)

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 500


# function to train the model
def train():
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # # progress update after every 50 batches.
        # if step % 50 == 0 and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate():
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []
valid_accuracy = []
# for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # train model
    train_loss, _ = train()

    # evaluate model
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    with torch.no_grad():
        preds = model(val_seq.to(device), val_mask.to(device))
        # train_preds = model(train_seq.to(device), train_mask.to(device))
        preds = preds.detach().cpu().numpy()
        # train_preds = train_preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        # train_preds = np.argmax(train_preds, axis=1)

    acc = accuracy_score(val_y, preds)
    # train_acc = accuracy_score(train_y, train_preds)
    # train_accuracy.append(train_acc)
    valid_accuracy.append(acc)
    print(
        f'Training Loss: {train_loss:.3f}\t Validation Loss: {valid_loss:.3f} \t vall acc: {acc:.3f}')

    # if acc > 0.92:
    #     break

    # get predictions for test data

with torch.no_grad():
    preds = model(val_seq.to(device), val_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
print(f'vall acc: {accuracy_score(val_y, preds):.3f}')

results = pd.DataFrame(zip(train_losses, valid_losses, valid_accuracy), columns=['train loss', 'val loss', 'val acc'])
results.to_csv('roberta_history.csv')

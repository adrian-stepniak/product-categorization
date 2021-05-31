import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from .utils import get_pretrained_model
import numpy as np


class BertClassifierFromPretrained(nn.Module):

    def __init__(self, model_dir, n_classes, freeze_model=True, device=None):
        super(BertClassifierFromPretrained, self).__init__()

        self.bert = get_pretrained_model(model_dir=model_dir, freeze_model=freeze_model)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, n_classes)

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

    # function to train the model
    def _train_step(self, train_dataloader, loss_func, optimizer):
        self.train()

        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds = []

        total_steps = len(train_dataloader)
        # iterate over batches
        for step, batch in enumerate(train_dataloader):

            # push the batch to gpu
            batch = [r.to(self.device) for r in batch]

            sent_id, mask, labels = batch

            self.zero_grad()
            preds = self(sent_id, mask)

            # compute the loss between actual and predicted values
            loss = loss_func(preds, labels)
            total_loss = total_loss + loss.item()

            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()

            # append the model predictions
            total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def _evaluate_step(self, val_dataloader, loss_func):
        self.eval()

        total_loss, total_accuracy = 0, 0
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(val_dataloader):
            # push the batch to gpu
            batch = [t.to(self.device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():
                # model predictions
                preds = self(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = loss_func(preds, labels)

                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader)

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def fit(self, train_dataloader, y_train, test_dataloader, y_test, epochs, loss_function, optimizer,
            save_best_model=True):

        # empty lists to store training and validation loss and accuracy of each epoch
        best_valid_loss = float('inf')
        train_losses = []
        valid_losses = []
        train_accuracy = []
        valid_accuracy = []
        # for each epoch
        for epoch in range(epochs):

            print(f'\n Epoch {epoch + 1} / {epochs}')

            # train model
            train_loss, train_pred = self._train_step(train_dataloader, loss_function, optimizer)
            train_pred = np.argmax(train_pred, axis=1)
            train_acc = accuracy_score(y_train, train_pred)

            # evaluate model
            valid_loss, val_pred = self._evaluate_step(test_dataloader, loss_function)
            val_pred = np.argmax(val_pred, axis=1)
            val_acc = accuracy_score(y_test, val_pred)

            # save the best model

            if save_best_model and (valid_loss < best_valid_loss):
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'saved_weights.pt')

            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracy.append(train_acc)
            valid_accuracy.append(val_acc)
            print(f'Training Loss: {train_loss:.3f}\t Validation Loss: {valid_loss:.3f}')
            print(f'Training acc: {train_acc:.3f}\t Validation acc: {val_acc:.3f}')

        results = pd.DataFrame(
            zip(train_losses, valid_losses, train_accuracy, valid_accuracy),
            columns=['train_loss', 'val_loss', 'train_acc', 'val_acc']
        )

        return results

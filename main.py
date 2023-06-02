import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc

import torch
import torch.nn as nn
import torch.optim as optim

from data import Dataset
from lstm import LSTMClassifier
from mylstm import MyLSTMClassifier
from transformer import TransformerClassifier

device = torch.device("cpu")
lstm_hidden_dim = 8
batch_size = 128
transformer_head = 2
transformer_hidden = 8
transformer_enclayers = 2
transformer_declayers = 2

def train_lstm(dataset):
    dataset.set_train()

    model = LSTMClassifier(hidden_dim=lstm_hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    model.train()

    for i in range(epochs):
        end_flag = False
        print('[epoch: {:0>4}/{:0>4}]'.format(i + 1, epochs))
        while end_flag is not True:
            optimizer.zero_grad()

            (data, label), end_flag = dataset.get_batch()
            data, label = data.to(device), label.to(device)
            pred = model(data)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        test_lstm(dataset, model, test=False)

    print('Test: ')
    test_lstm(dataset, model, test=True)
    torch.save(model.state_dict(), 'checkpoints/lstm.pth')


def test_lstm(dataset, model=None, test=True, load_ckpt=False):
    if test:
        dataset.set_test()
    else:
        dataset.set_train()

    if model is None:
        model = LSTMClassifier(hidden_dim=lstm_hidden_dim)
    if load_ckpt:
        model.load_state_dict(torch.load('checkpoints/lstm.pth'))

    model.eval()
    end_flag = False

    prob_all = []
    label_all = []
    while end_flag is not True:
        (data, label), end_flag = dataset.get_batch()
        data, label = data.to(device), label.to(device)
        pred = model(data)
        pred_label = torch.argmax(pred, dim=1)

        prob_all.extend(pred_label.cpu().numpy())
        label_all.extend(label.cpu().numpy())

    print('accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score:{:.4f}, auc: {:.4f}'.\
        format(accuracy_score(label_all, prob_all),
               precision_score(label_all, prob_all),
               recall_score(label_all, prob_all),
               f1_score(label_all, prob_all),
               roc_auc_score(label_all, prob_all)))
    if test:
        fpr, tpr, thresholds = roc_curve(label_all, prob_all)
        print('fpr: ', fpr)
        print('tpr: ', tpr)
        plt.plot(fpr, tpr, '--', label='area = {0:.2f}'.format(auc(fpr, tpr)))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.savefig('./img/lstm_roc.png')


def train_mylstm(dataset):
    dataset.set_train()

    model = MyLSTMClassifier(nhid=lstm_hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    model.train()

    for i in range(epochs):
        end_flag = False
        print('[epoch: {:0>4}/{:0>4}]'.format(i + 1, epochs))
        while end_flag is not True:
            optimizer.zero_grad()

            (data, label), end_flag = dataset.get_batch()
            data, label = data.to(device), label.to(device)
            pred = model(data)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        test_mylstm(dataset, model, test=False)

    print('Test: ')
    test_mylstm(dataset, model, test=True)
    torch.save(model.state_dict(), 'checkpoints/mylstm.pth')


def test_mylstm(dataset, model=None, test=True, load_ckpt=False):
    if test:
        dataset.set_test()
    else:
        dataset.set_train()

    if model is None:
        model = MyLSTMClassifier(hidden_dim=lstm_hidden_dim)
    if load_ckpt:
        model.load_state_dict(torch.load('checkpoints/mylstm.pth'))

    model.eval()
    end_flag = False

    prob_all = []
    label_all = []
    while end_flag is not True:
        (data, label), end_flag = dataset.get_batch()
        data, label = data.to(device), label.to(device)
        pred = model(data)
        pred_label = torch.argmax(pred, dim=1)

        prob_all.extend(pred_label.cpu().numpy())
        label_all.extend(label.cpu().numpy())

    print('accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score:{:.4f}, auc: {:.4f}'.\
        format(accuracy_score(label_all, prob_all),
               precision_score(label_all, prob_all),
               recall_score(label_all, prob_all),
               f1_score(label_all, prob_all),
               roc_auc_score(label_all, prob_all)))
    if test:
        fpr, tpr, thresholds = roc_curve(label_all, prob_all)
        print('fpr: ', fpr)
        print('tpr: ', tpr)
        plt.plot(fpr, tpr, '--', label='area = {0:.2f}'.format(auc(fpr, tpr)))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.savefig('./img/mylstm_roc.png')


def train_transformer(dataset):
    dataset.set_train()

    model = TransformerClassifier(nhead=transformer_head, nhid=transformer_hidden, nenclayers=transformer_enclayers, ndeclayers=transformer_declayers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    model.train()

    for i in range(epochs):
        end_flag = False
        print('[epoch: {:0>4}/{:0>4}]'.format(i + 1, epochs))
        while end_flag is not True:
            optimizer.zero_grad()

            (data, label), end_flag = dataset.get_batch()
            data, label = data.to(device), label.to(device)
            pred = model(data)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        test_lstm(dataset, model, test=False)

    print('Test: ')
    test_transformer(dataset, model, test=True)
    torch.save(model.state_dict(), 'checkpoints/transformer.pth')


def test_transformer(dataset, model=None, test=True, load_ckpt=False):
    if test:
        dataset.set_test()
    else:
        dataset.set_train()

    if model is None:
        model = TransformerClassifier(nhead=transformer_head, nhid=transformer_hidden, nenclayers=transformer_enclayers, ndeclayers=transformer_declayers)
    if load_ckpt:
        model.load_state_dict(torch.load('checkpoints/transformer.pth'))

    model.eval()
    end_flag = False

    prob_all = []
    label_all = []
    while end_flag is not True:
        (data, label), end_flag = dataset.get_batch()
        data, label = data.to(device), label.to(device)
        pred = model(data)
        pred_label = torch.argmax(pred, dim=1)

        prob_all.extend(pred_label.cpu().numpy())
        label_all.extend(label.cpu().numpy())

    print('accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score:{:.4f}, auc: {:.4f}'.\
        format(accuracy_score(label_all, prob_all),
               precision_score(label_all, prob_all),
               recall_score(label_all, prob_all),
               f1_score(label_all, prob_all),
               roc_auc_score(label_all, prob_all)))
    if test:
        fpr, tpr, thresholds = roc_curve(label_all, prob_all)
        print('fpr: ', fpr)
        print('tpr: ', tpr)
        plt.plot(fpr, tpr, '--', label='area = {0:.2f}'.format(auc(fpr, tpr)))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.savefig('./img/transformer_roc.png')


if __name__ == '__main__':
    torch.manual_seed(3407)
    dataset = Dataset(batch_size=batch_size, shuffle=True)
    train_transformer(dataset)
    # train_mylstm(dataset)
    # train_lstm(dataset)
    # test_lstm(dataset, model=None, load_ckpt=True)

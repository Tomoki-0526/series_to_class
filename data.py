import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import random


class Dataset():
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        train_X, train_y, test_X, test_y = self.data_preprocessing()
        self.train_X, self.train_y, self.test_X, self.test_y = \
            torch.Tensor(train_X), torch.Tensor(train_y).long(), torch.Tensor(test_X), torch.Tensor(test_y).long()

        if shuffle:
            idx = [i for i in range(len(self.train_y))]
            idx = random.shuffle(idx)
            self.train_X = self.train_X[idx].squeeze()
            self.train_y = self.train_y[idx].squeeze()

        self.train_flag = True
        self.test_flag = False

        self.train_ptr = 0
        self.test_ptr = 0

    def data_preprocessing(self):
        train_path = 'dataset/train.csv'
        test_path = 'dataset/test.csv'
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        features = ['heartrate', 'resprate', 'map', 'o2sat']
        # 删除缺值行
        train_data = train_data.dropna(axis=0, how='any')
        test_data = test_data.dropna(axis=0, how='any')
        # 去除时间戳
        train_data = train_data.drop('time', axis=1)
        test_data = test_data.drop('time', axis=1)
        # 标准化
        scaler = StandardScaler()
        train_data[features] = scaler.fit_transform(train_data[features])
        test_data[features] = scaler.transform(test_data[features])

        padding_num = -100
        train_mat = train_data.values
        train_x_list = []
        train_y_list = []
        previous_id = -1
        for i in range(len(train_mat)):
            if train_mat[i, 0] == previous_id:
                train_x_list[-1].append(train_mat[i, 1:5].tolist())
            else:
                train_x_list.append([train_mat[i, 1:5].tolist()])
                train_y_list.append(train_mat[i, -1])
                previous_id = train_mat[i, 0]

        max_len = -1
        for i in range(len(train_x_list)):
            if len(train_x_list[i]) > max_len:
                max_len = len(train_x_list[i])
        max_len += 1

        for i in range(len(train_x_list)):
            while len(train_x_list[i]) < max_len:
                train_x_list[i].append([padding_num for i in range(4)])

        train_X = np.array(train_x_list)
        train_y = np.array(train_y_list)

        test_mat = test_data.values
        test_x_list = []
        test_y_list = []
        previous_id = -1
        for i in range(len(test_mat)):
            if test_mat[i, 0] == previous_id:
                test_x_list[-1].append(test_mat[i, 1:5].tolist())
            else:
                test_x_list.append([test_mat[i, 1:5].tolist()])
                test_y_list.append(test_mat[i, -1])
                previous_id = test_mat[i, 0]

        for i in range(len(test_x_list)):
            while len(test_x_list[i]) < max_len:
                test_x_list[i].append([padding_num for k in range(4)])

        test_X = np.array(test_x_list)
        test_y = np.array(test_y_list)

        return train_X, train_y, test_X, test_y

    def set_train(self):
        self.train_flag = True
        self.test_flag = False
        self.train_ptr = 0
        self.test_flag = 0

    def set_test(self):
        self.train_flag = False
        self.test_flag = True
        self.train_ptr = 0
        self.test_ptr = 0

    def get_batch(self):
        end_flag = False
        if self.train_flag:
            if self.train_ptr + self.batch_size <= len(self.train_X):
                ret = (self.train_X[self.train_ptr : self.train_ptr + self.batch_size], \
                    self.train_y[self.train_ptr : self.train_ptr + self.batch_size])
                self.train_ptr += self.batch_size
            else:
                ret = (self.train_X[self.train_ptr :], self.train_y[self.train_ptr :])
                end_flag = True
                self.train_ptr = 0
        elif self.test_flag:
            if self.test_ptr + self.batch_size <= len(self.test_X):
                ret = (self.test_X[self.test_ptr : self.test_ptr + self.batch_size], \
                    self.test_y[self.test_ptr : self.test_ptr + self.batch_size])
                self.test_ptr += self.batch_size
            else:
                ret = (self.test_X[self.test_ptr :], self.test_y[self.test_ptr :])
                end_flag = True
                self.test_ptr = 0
        else:
            assert False
        return ret, end_flag


if __name__ == '__main__':
    # data_preprocessing()
    dataset = Dataset()

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_pre_mean) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_pre_mean = batch_pre_mean.float().to(self.device)
                outputs = self.model(batch_x, batch_pre_mean, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_pre_mean) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_pre_mean = batch_pre_mean.float().to(self.device)

                outputs = self.model(batch_x, batch_pre_mean, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                # loss = 0.1*criterion(outputs[:,:], batch_x[:,:])+criterion(outputs[:,-80:], batch_x[:,-80:])
                loss = criterion(outputs[:, :], batch_x[:, :])
                # a = self.kl_divergence(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        thre_data, thre_loader = self._get_data(flag='test')
        # test_data, test_loader = self._get_data(flag='thre')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        # preds_train = []
        train_subsequence_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_pre_mean) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_pre_mean = batch_pre_mean.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, batch_pre_mean, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                # 计算每个子序列的平均得分
                train_avg_scores = np.mean(score, axis=1, keepdims=True)
                attens_energy.append(score)
                train_subsequence_energy.append(train_avg_scores)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        train_subsequence_energy = np.concatenate(train_subsequence_energy, axis=0).reshape(-1)
        train_subsequence_energy_score = np.array(train_subsequence_energy)
        # (2) find the threshold
        attens_energy = []
        test_labels = []
        test_subsequence_energy = []
        pred_data = None
        preds = []
        trues = []
        for i, (batch_x, batch_y, batch_pre_mean) in enumerate(thre_loader):
            batch_x = batch_x.float().to(self.device)
            batch_pre_mean = batch_pre_mean.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, batch_pre_mean, None, None)
            pred_data = outputs
            pred_data = pred_data[:, -self.args.pred_len:, 0:]
            pred_data = pred_data.detach().cpu().numpy()
            pred = pred_data
            preds.append(pred)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            # 计算每个子序列的平均得分
            test_avg_scores = np.mean(score, axis=1, keepdims=True)
            test_subsequence_energy.append(test_avg_scores)
            attens_energy.append(score)
            test_labels.append(batch_y[:, :])

        import pandas as pd
        # 使用 concatenate 函数将它们合并
        merged_array = np.concatenate(preds, axis=0)
        preds = np.array(merged_array)
        merged_data1 = preds.reshape(-1, 1)
        # 创建 StandardScaler 对象
        scaler = StandardScaler()
        data = pd.read_csv(os.path.join('./dataset/Oil_anomal_detection_1', 'train.csv'))
        data = data.drop(columns=['date'])
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        scaler.fit(data)
        # 使用 inverse_transform 方法进行反归一化
        # original_data_subset = scaler.inverse_transform(merged_data1)
        original_data_subset = merged_data1
        df = pd.DataFrame(original_data_subset)
        df.to_csv('./dataset/Oil_anomal_detection_1/' + self.args.model + '_' + 'pred_data.csv',
                                index_label=None)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_subsequence_energy = np.concatenate(test_subsequence_energy, axis=0).reshape(-1)
        test_subsequence_energy_score = np.array(test_subsequence_energy)
        # combined_energy = np.concatenate([train_subsequence_energy_score, test_subsequence_energy_score], axis=0)
        combined_energy = np.concatenate([test_subsequence_energy_score], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        ## anomaly_ratio
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_subsequence_energy_score > threshold).astype(int)

        # 评估测试集
        all_pred_labels = []
        with torch.no_grad():
            for batch_x, _, batch_pre_mean in thre_loader:
                batch_x = batch_x.float().to(self.device)
                batch_pre_mean = batch_pre_mean.float().to(self.device)
                outputs = self.model(batch_x, batch_pre_mean, None, None)
                scores = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1).mean(dim=1)
                # 根据阈值生成标签
                batch_labels = np.where(scores.detach().cpu().numpy() > threshold, 1, 0)
                print('xx', batch_labels.shape)
                # 扩展每个子序列标签为整个子序列
                expanded_labels = np.repeat(batch_labels, 120)
                # print('aa', expanded_labels.shape)
                # expanded_labels = expanded_labels[:, -30:]
                all_pred_labels.extend(expanded_labels)

        pred = np.array(all_pred_labels)
        # 定义初始窗口大小和滑动步长
        window_size = 120
        stride = 30
        # 对新的一维numpy数组按照120大小的窗口和30为滑动步长再次划分子序列
        pred_subsequences_new = [pred[i:i+window_size] for i in range(0, len(pred), window_size)]
        abc = np.array(pred_subsequences_new)
        # 取第一个子序列前90个数据，其他子序列取最后30个数据
        # 取第一个子序列前90个数据，其他子序列取最后30个数据
        pred_labels_new = pred_subsequences_new[0]
        for subsequence in pred_subsequences_new[1:]:
            pred_labels_new = np.concatenate((pred_labels_new, subsequence[-30:]))
        # 拼接所有子序列为一个新的numpy数组
        # final_data = np.concatenate(a)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        test_subsequences_new = [test_labels[i:i + window_size] for i in range(0, len(pred), window_size)]
        test_labels_new = test_subsequences_new[0]
        for subsequence in test_subsequences_new[1:]:
            test_labels_new = np.concatenate((test_labels_new, subsequence[-30:]))
        test_labels = np.array(test_labels_new)
        gt = test_labels.astype(int)
        pred_labes = np.array(pred_labels_new)
        pred = pred_labes.astype(int)
        print(batch_x.shape)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        # gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)


        import pandas as pd
        DA = {'pred': pred.flatten(), 'true': gt.flatten()}
        pd.DataFrame(DA).to_csv('./dataset/Oil_anomal_detection_1/' + self.args.model + '_' + 'pred.csv', index_label=None)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return

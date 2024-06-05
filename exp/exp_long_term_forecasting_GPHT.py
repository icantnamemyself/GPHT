from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_GPHT(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_GPHT, self).__init__(args)
        assert args.pred_len == args.token_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, data=None):
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        itrs = self.args.pred_len // self.args.token_len
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.float().to(self.device)

                # 还有问题
                batch_x_mark = batch_y_mark = dec_inp = None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                outputs = outputs[:, :, f_dim:]
                batch_y = torch.cat(
                    [batch_x[:, self.args.token_len:, :], batch_y[:, -self.args.token_len:, f_dim:].to(self.device)],
                    dim=1)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        iter_verbose = 1000
        if self.args.load_pretrain:
            iter_verbose = 100
            print('loading')
            setting2 = setting.replace(self.args.data, 'pretrain')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting2, 'checkpoint.pth', ),
                           map_location=self.device))
            param_train = 0
            param_all = 0
            for name, param in self.model.named_parameters():
                if 'forecast_head' in name:
                    print(name)
                    param_train += param.numel()
                    param_all += param.numel()
                else:
                    param.requires_grad = False
                    param_all += param.numel()
            print(
                f'trainable parameters num: {clever_format(param_train)}, all parameters num: {clever_format(param_all)},'
                f'ratio: {param_train / param_all * 100} %')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.float().to(self.device)

                batch_x_mark = batch_y_mark = dec_inp = None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, :, f_dim:]
                batch_y = torch.cat(
                    [batch_x[:, self.args.token_len:, :], batch_y[:, -self.args.token_len:, f_dim:].to(self.device)],
                    dim=1)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % iter_verbose == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            _, test_loss = self.test(setting)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        print(torch.cuda.device_count())
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='ar_test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth', ), map_location=self.device))

        itrs = int(np.ceil(self.args.ar_pred_len / self.args.token_len))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                pred_tmp = []
                batch_x_backup = batch_x
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.ar_pred_len:, :].to(self.device)
                batch_y2 = batch_y.detach().cpu().numpy()
                batch_y2 = batch_y2[:, :, f_dim:]
                true = batch_y2
                batch_x = batch_x.float().to(self.device)
                for j in range(itrs):
                    if not j:
                        batch_x = batch_x.float().to(self.device)
                    else:
                        batch_x = torch.cat([batch_x, pred_tmp[-1]], dim=1)[:, -self.args.seq_len:, :]

                    batch_x_mark = batch_y_mark = dec_inp = None
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                    pred_tmp.append(outputs[:, -self.args.token_len:, f_dim:])

                pred = torch.cat(pred_tmp, dim=1).detach().cpu().numpy()[:, :self.args.ar_pred_len, :]
                # pred = self.model.multi_step_forecast(batch_x, itrs).detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                # if i % 1 == 0:
                #     # print(np.average(loss_san), np.average(loss_token))
                #     input = batch_x_backup.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                #     pd = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.svg'))

        preds = np.array(preds, dtype=object)
        trues = np.array(trues, dtype=object)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return mae, mse

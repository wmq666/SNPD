import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from tqdm import tqdm
from user import make_path, SNR, SAD, tensor_to_numpy


class TestNr2N:
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.gpu_num = args.gpu_num
        self.test_name = args.test_name
        self.pretrain_model = args.pretrain_model
        self.num_groups_test = args.num_groups_test
        self.save_root = make_path('../Result', self.test_name)
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')
        self.pretrained_model_path = make_path(self.save_root, 'output/model_{}.pth'.format(self.pretrain_model))
        self.wn = np.load('wn.npy')

    def test(self):
        model = torch.load(self.pretrained_model_path, map_location=self.device)
        model.eval()
        clean_data = np.load(make_path(self.save_root, 'test_dataset/test_gt.npy'))
        noisy_data = np.load(make_path(self.save_root, 'test_dataset/test_noisy.npy'))
        noisier_data = np.load(make_path(self.save_root, 'test_dataset/test_noisier.npy'))
        prediction_data, p_SNR, p_SAD, i_SNR, i_SAD, middle_data = [], [], [], [], [], []
        for i in tqdm(range(self.num_groups_test), desc='test', ncols=100):
            clean = clean_data[i, :].reshape(1024,)
            noisy = noisy_data[i, :].reshape(1024,)
            noisier = noisier_data[i, :].reshape(1024,)
            noisier = torch.tensor(noisier).reshape(1, 1, 1024)
            noisier = noisier.type(torch.FloatTensor).to(self.device)
            middle = model(noisier)
            prediction = ((1 + self.alpha**2) * model(noisier) - noisier) / (self.alpha**2)
            prediction = tensor_to_numpy(prediction)
            prediction = prediction.reshape(1024,)
            middel = tensor_to_numpy(middle)
            middle = middel.reshape(1024,)
            prediction_data.append(prediction)
            middle_data.append(middle)
            predict_SNR = round(SNR(clean, prediction), 4)
            predict_SAD = round(SAD(clean, prediction), 4)
            initial_SNR = round(SNR(clean, noisy), 4)
            initial_SAD = round(SAD(clean, noisy), 4)
            p_SNR.append(predict_SNR)
            p_SAD.append(predict_SAD)
            i_SNR.append(initial_SNR)
            i_SAD.append(initial_SAD)
        np.save(make_path(self.save_root, 'output/ss_data.npy'), (p_SNR, i_SNR, p_SAD, i_SAD))
        np.save(make_path(self.save_root, 'output/prediction_data.npy'), np.array(prediction_data).reshape(self.num_groups_test, 1024))
        np.save(make_path(self.save_root, 'output/middle_data.npy'), np.array(middle_data).reshape(self.num_groups_test, 1024))

    def plot(self, mode):
        if mode == 'on':
            clean_data = np.load(make_path(self.save_root, 'test_dataset/test_gt.npy'))
            noisy_data = np.load(make_path(self.save_root, 'test_dataset/test_noisy.npy'))
            prediction_all = np.load(make_path(self.save_root, 'output', f'{self.noise_std:.2f}', 'prediction_data.npy'))
            for i in tqdm(range(self.num_groups_test), desc='plot', ncols=100):
                clean = clean_data[i, :].reshape(1024,)
                noisy = noisy_data[i, :].reshape(1024,)
                prediction = prediction_all[i, :].reshape(1024,)
                plt.figure(figsize=(6, 5))
                plt.plot(self.wn, noisy, linestyle='solid', color='#CCCCCC', label='noisy')
                plt.plot(self.wn, clean, linestyle='solid', color='#CC0000', label='gt')
                plt.plot(self.wn, prediction, linestyle='solid', color='#0000FF', label='denoised')
                plt.xlabel('Wavenumber (cm\u207B\u00B9)', fontsize=15)
                plt.ylabel('Intensity(a.u.)', fontsize=15)
                plt.legend(loc='upper right')
                plt.rcParams['font.sans-serif'] = ['Times New Roman']
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout()
                plt.savefig(make_path(self.save_root, 'output/pic/{}.png'.format(i)), dpi=300)
                plt.clf()
                plt.close()

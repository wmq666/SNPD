import torch
from torch import nn
from tqdm import tqdm
from model import UNet
from torch import optim
from user import make_path
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import nr2n_dataset


class TrainNr2N:
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.alpha = args.alpha
        self.gpu_num = args.gpu_num
        self.n_epochs = args.n_epochs
        self.threshold = args.threshold
        self.test_name = args.test_name
        self.batch_size = args.batch_size
        self.n_snapshot = args.n_snapshot
        self.save_root = make_path('../Result', self.test_name)
        self.train_dataset = nr2n_dataset(make_path(self.save_root, 'train_dataset/train_noisy.npy'),
                                          make_path(self.save_root, 'train_dataset/train_noisier.npy'), 1024)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')
        self.model = UNet(1, 1).to(self.device)
        self.criterion_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99))
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0.05, total_iters=20)

    def train(self):
        print(self.device, '----------Nr2N----------')
        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for data in tepoch:
                    self.model.train()
                    self.optimizer.zero_grad()

                    noisy, noisier = data
                    noisy, noisier = noisy.to(self.device), noisier.to(self.device)

                    prediction = self.model(noisier)
                    loss = self.criterion_mse(prediction, noisy)
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(rec_loss=loss.item())
                self.scheduler.step()

            # Checkpoints
            if epoch >= self.threshold:
                if epoch % self.n_snapshot == 0 or epoch == self.n_epochs:
                    torch.save(self.model, make_path(self.save_root, 'output/model_{}.pth'.format(epoch)))


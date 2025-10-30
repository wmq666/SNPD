import argparse

import numpy as np

from train import TrainNr2N
from test import TestNr2N


# Arguments
parser = argparse.ArgumentParser(description='Train Nr2N public')
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--num_groups_test', default=1000, type=int)
parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--n_snapshot', default=1, type=int, help='model checkpoint interval')
parser.add_argument('--threshold', default=20, type=int, help='model checkpointing start epoch')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--pretrain_model', default=20, type=int, help='which pre-trained model to use')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--test_name', default='test_1', type=str)
args = parser.parse_args()


train_Nr2N = TrainNr2N(args=args)
train_Nr2N.train()
test_Nr2N = TestNr2N(args=args)
test_Nr2N.test()
test_Nr2N.plot('off')
#
# a = np.mean(np.load('../Result/test_1/output/ss_data.npy'), axis=1)[0]

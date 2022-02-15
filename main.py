from data import utils
import torch
import numpy as np
import Model
import torch.optim as optim
import argparse
import time
import metrics
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch SNPAD')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=20,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=1,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=98765,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

device = torch.device("cuda")

dataset = 'thyroid'

trainLoader, testLoader, x_train, x_test, y_train, y_test = utils.getDataLoader(file=dataset,
                                                                                batchsize=args.batch_size)
O = torch.cat((x_train, y_train), dim=1)

shape = x_train.shape
data_dim = [shape[1], 600, 200]


class TrainSADNP():

    def __init__(self):
        super(TrainSADNP, self).__init__()
        self.model = Model.SNPAD(data_dim).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.labeled_ratio = 0.2

    def train(self):
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            loss = self.trainPerEpoch()
            print('| end of epoch {:3d} | time: {:4.2f}s | train loss {:4.2f}'.format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      loss))

    def trainPerEpoch(self):
        self.model.train()
        train_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(trainLoader):
            self.optimizer.zero_grad()
            num = int(self.labeled_ratio * batch_x.shape[0])
            labeled_data = batch_x[0:num, :]
            label = batch_y[0:num, :]
            unlabeled_data = batch_x[num:batch_x.shape[0], :]
            loss = self.model(unlabel_data=unlabeled_data, label_data=labeled_data, label=label, O=O,
                              isTraining=True)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            label = np.array([]).reshape(-1, 1)
            y_predict = np.array([]).reshape(-1, 1)
            for step, (batch_x, batch_y) in enumerate(testLoader):
                predictions = self.model(unlabel_data=batch_x, label_data=batch_x, label=batch_y, O=O,
                                         isTraining=False)
                # mean value
                pres = torch.unsqueeze(torch.mean(predictions, dim=1), dim=1)
                y_predict = np.concatenate((y_predict, pres.cpu().numpy()), axis=0)
                label = np.concatenate((label, batch_y.cpu().numpy()), axis=0)
            auc_roc = metrics.auc_roc(y_predict=y_predict, y_true=label)
            recall = metrics.recall(y_predict=y_predict, y_true=label)
        return auc_roc, recall

    def run(self):
        self.train()
        auc, recall = self.evaluate()
        print(auc)


if __name__ == '__main__':
    TrainSADNP().run()

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
import torch.optim as optim

import os
import time
import shutil
import pickle
import time

from tqdm import tqdm
from utils import AverageMeter
from model import IsoNN
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, fold, data_loader, test_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """ 
        self.result_dir = config.result_dir
        self.config = config
        self.fold_count = fold

        self.k = 4
        self.c = 3
        self.num_node = config.num_node
        self.feature_size = ((config.num_node - self.k) + 1) ** 2 * self.c
        self.n_hidden1 = config.hidden1
        self.n_hidden2 = config.hidden2
        self.nclass = config.nclass
       
        self.batch_size = config.batch_size
        self.M = config.M

        self.train_loader = data_loader
        self.num_train = len(self.train_loader.dataset)
        
        self.test_loader = test_loader
        self.num_test = len(self.test_loader.dataset)
        

        self.epochs = 80#config.epochs
        self.start_epoch = 0
       
        self.lr = config.init_lr

     
        

        self.model_name = 'IsoNN_k1_{}_c1_{}'.format(
            self.k, self.c
        )

        # build RAM model
        self.model = IsoNN(self.k, self.c, self.feature_size, self.n_hidden1, self.n_hidden2, self.nclass)
       
      
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-3, #weight_decay=1e-5
        )


    def train(self):
        loss = []
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            loss.append(train_loss)

        
            msg = "train loss: {:.3f} - train acc: {:.3f} "
            print(msg.format(train_loss, train_acc))
            
        elapsed = time.time() - start
        # results = {'k': self.k, 'c': self.c, 'loss': loss, 'time': elapsed}
        # data_name = self.config.data_dir.split('/')[-1]
        # f = open(self.result_dir + data_name+'_IsoLayer_fold_'+str(self.fold_count), 'wb')
        # f = open(self.result_dir+data_name+'_FastIsoLayer_fold_'+str(self.fold_count), 'wb')
        # pickle.dump(results, f)
        # f.close()
        # print('c', self.c, 'k', self.k)


    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        f1s = AverageMeter()
        recalls = AverageMeter()
        aucs = AverageMeter()
        precisions = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader): # x [B, W, H], y [B]
                x, y = Variable(x), Variable(y)

                probas = self.model(x)
                
                predicted = torch.max(probas, 1)[1]

                loss = F.nll_loss(probas, y) # supervised loss
                acc = accuracy_score(y, predicted)
                f1 = f1_score(y, predicted)
                prec = precision_score(y, predicted)
                rec = recall_score(y, predicted)
                
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc)
                f1s.update(f1)
                recalls.update(rec)
                precisions.update(prec)
                
                # compute gradients and update SGD
                self.optimizer.zero_grad()
             
                loss.backward()

                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.update(self.batch_size)

       
        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        accs = AverageMeter()
        f1s = AverageMeter()
        recalls = AverageMeter()
        aucs = AverageMeter()
        precisions = AverageMeter()
       
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = Variable(x), Variable(y)

                # duplicate 10 times
                x = x.repeat(self.M, 1, 1, 1)
                probas = self.model(x)
                
                probas = probas.view(
                    self.M, -1, probas.shape[-1]
                )
                probas = torch.mean(probas, dim=0)

                predicted = probas.data.max(1, keepdim=True)[1]


                acc = accuracy_score(y, predicted)
                f1 = f1_score(y, predicted)
                prec = precision_score(y, predicted)
                rec = recall_score(y, predicted)
                
                accs.update(acc)
                f1s.update(f1)
                recalls.update(rec)
                precisions.update(prec)
                
              

        print('k, c', self.k, self.c)
        print('Accuracy:', accs.avg)
        print('F1: ', f1s.avg)
        print('Precision: ', precisions.avg)
        print('Recall: ', recalls.avg)

        result = [accs.avg, f1s.avg, precisions.avg, recalls.avg]
        return result


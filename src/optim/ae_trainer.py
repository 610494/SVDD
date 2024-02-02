from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
import wandb
import random


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=4, factor=0.85, threshold=0.00001)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in tqdm(range(self.n_epochs)):

            loss_epoch = 0.0
            n_batches = 0

            epoch_start_time = time.time()
            for data in train_loader:
                # inputs, _, _, _, _ = data
                inputs, _ = data
                inputs = inputs.to(self.device).float()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                print(f"************\n[optim/ae_trainer.py] Input size before forward pass: {inputs.size()}\n************") # Input size before forward pass: torch.Size([128, 1, 20])
                outputs = ae_net(inputs)
                print(f"************\n[optim/ae_trainer.py] Output size after forward pass: {outputs.size()}\n************") # Output size after forward pass: torch.Size([128, 1, 10])
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step(loss_epoch / n_batches / self.batch_size)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches / self.batch_size))
            wandb.log({"ae_train/loss": loss_epoch /
                      n_batches / self.batch_size})

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(
            batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx, classes, filename = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2,
                                   dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            classes,
                                            filename
                                            ))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores, _, _ = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')

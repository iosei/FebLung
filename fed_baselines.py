import copy
import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import tqdm

from utils import *
import torch.nn.functional as F
import sys
sys.path.append('C:/Users/NEW/Desktop/FedLung/FedLung/utils')
from fed_utils import *
import sys
sys.path.append('C:/Users/NEW/Desktop/FedLung/FedLung/utils')
from utils.dataloader import HospitalDatasetSingle
from collections import  Counter
import  random
from opacus import PrivacyEngine
import warnings

torch.manual_seed(0)

class FedTrain(object):
    def __init__(self, args, dataset=None, client_data=None):
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset_list = dataset
        self.client_data = client_data
        self.ldr_train = DataLoader(HospitalDatasetSingle(self.dataset_list, self.client_data), batch_size=self.args.local_bs, shuffle=True)
        self.server_params = None #copy.deepcopy(list(self.net.parameters()))

    from opacus import PrivacyEngine

    def train_client(self, netC):
        self.server_params = copy.deepcopy(list(netC.parameters()))
        netC.train()

        # Initialize optimizer
        optimizer = torch.optim.Adam(netC.parameters(), lr=self.args.lr)

        # Apply differential privacy with PrivacyEngine if enabled
        if self.args.use_dp:
            # Suppress the warning for now if you prefer
            warnings.filterwarnings("ignore", category=UserWarning)

            # Initialize the Privacy Engine with secure mode enabled
            privacy_engine = PrivacyEngine(secure_mode=False)

            # Make sure to provide a valid data loader, optimizer, and model
            # Unpack without using the privacy_engine
            privacy_engine = PrivacyEngine()

            # Attach the Privacy Engine to your model and optimizer
            privacy_engine.attach(netC)  # Attach the privacy engine to the model
            optimizer = privacy_engine.make_private(
                optimizer=optimizer,
                data_loader=self.ldr_train,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
            )

            # Attach privacy engine to optimizer


            print(f"Differential Privacy enabled with noise multiplier {self.args.noise_multiplier}")

        epoch_loss = []
        epoch_acc = []

        # Local epochs
        for iter in range(self.args.local_ep):
            batch_loss = []
            train_correct = 0
            train_total = 0

            # Loop through the training batches
            for images, labels in self.ldr_train:
                images = images.to(self.args.device).float()
                labels = labels.to(self.args.device).long()

                netC.zero_grad()
                log_probs = netC(images)
                _, predicted = torch.max(log_probs, 1)
                loss = self.criterion(log_probs, labels)

                # FedProx implementation
                if self.args.algo == "fedprox":
                    w_diff = torch.tensor(0., device=self.args.device)
                    mu = 0.001
                    for w, w_t in zip(self.server_params, netC.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += mu / 2. * w_diff

                loss.backward()
                optimizer.step()

                # Calculate batch accuracy
                train_total += labels.size(0)
                train_correct += torch.sum(predicted == labels.data)
                batch_accuracy = (train_correct.double() / train_total) * 100
                print(f"Batch Accuracy: {batch_accuracy:.2f}%")

                batch_loss.append(loss.item())

            # Calculate epoch metrics
            acc_avg = (train_correct.double() / train_total) * 100
            epoch_acc.append(acc_avg.cpu())
            loss_avg = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(loss_avg)

            print(f'Epoch [{iter + 1}/{self.args.local_ep}] - Loss: {loss_avg:.6f}, Accuracy: {acc_avg:.2f}%')

        return netC.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)


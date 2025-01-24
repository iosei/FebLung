import copy
import torch
import  numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils.dataloader import HospitalDatasetSingle

# from Pyfhel import Pyfhel, PyCtxt
#
# # Initialize homomorphic encryption context
# HE = Pyfhel()
# HE.contextGen(p=65537, m=4096)  # Generate context with a large prime modulus and polynomial degree
# HE.keyGen()  # Generate public and private keys
#
#
# def encrypt_weights(weights, HE):
#     encrypted_weights = {}
#     for k, v in weights.items():
#         encrypted_weights[k] = HE.encrypt(v.flatten())  # Flatten and encrypt each tensor
#     return encrypted_weights
#
#
# def decrypt_weights(encrypted_weights, HE):
#     decrypted_weights = {}
#     for k, v in encrypted_weights.items():
#         decrypted_weights[k] = HE.decrypt(v).reshape(weights[k].shape)  # Decrypt and reshape to original form
#     return decrypted_weights


# def FedAvgHE(w_locals, HE):
#     """
#     Federated Averaging using Homomorphic Encryption (HE).
#
#     Parameters:
#     w_locals: List of encrypted model weights from clients.
#     HE: The homomorphic encryption context.
#
#     Returns:
#     Decrypted and averaged model weights.
#     """
#     w_avg = copy.deepcopy(w_locals[0])
#
#     # Aggregate encrypted model updates
#     for k in w_avg.keys():
#         for i in range(1, len(w_locals)):
#             w_avg[k] += w_locals[i][k]
#         w_avg[k] = w_avg[k] / len(w_locals)
#
#     # Decrypt the aggregated weights
#     w_avg = decrypt_weights(w_avg, HE)
#
#     return w_avg
#
#
# # Example usage:
# # Encrypting client model weights
# encrypted_w_locals = [encrypt_weights(client_model.state_dict(), HE) for client_model in client_models]
#
# # Aggregating using FedAvg with homomorphic encryption
# global_model_weights = FedAvgHE(encrypted_w_locals, HE)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        #if k not in 'bn':
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

import torch
import torch.nn as nn
import copy
from torch.distributions import Laplace

import torch
import torch.nn as nn
import copy
from torch.distributions import Laplace

def add_laplace_noise(tensor, epsilon, sensitivity=0.2, is_training=True):
    if not is_training:
        return tensor  # No noise during testing

    scale = sensitivity / epsilon
    noise = Laplace(loc=0.0, scale=scale).sample(tensor.shape).to(tensor.device)
    return tensor + noise

def FedAvgDP(w, epsilon=0.00000000000001, sensitivity=0.2, round_num=1, add_noise_every=3, is_training=True):
    """
    Federated Averaging with Differential Privacy (DP) during training.
    - Only adds noise every few rounds to prevent too much accuracy loss.
    - Uses higher epsilon and lower sensitivity to balance privacy and performance.
    """
    w_avg = copy.deepcopy(w[0])

    # Aggregate model updates
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

        # Add Laplace noise only during training and only every `add_noise_every` rounds
        if is_training and (round_num % add_noise_every == 0):
            w_avg[k] = add_laplace_noise(w_avg[k], epsilon, sensitivity)

    return w_avg




# Example usage:
# w = [client_model_1.state_dict(), client_model_2.state_dict(), client_model_3.state_dict()]
# epsilon = 0.1  # Privacy budget
# sensitivity = 1.0  # Sensitivity of the model
# global_model = FedAvgDP(w, epsilon, sensitivity)

import torch
import torch.optim as optim
import copy

def FedOpt(w_locals, global_model, lr=1e-3):
    """
    FedOpt: Federated Optimization Algorithm
    This function aggregates the updates from multiple clients into a global model using Adam optimizer.

    Parameters:
    w_locals: list of state_dict from client models
    global_model: the global model to be updated
    lr: learning rate for Adam optimizer

    Returns:
    Updated global model state_dict.
    """
    # Start with the current global model's state_dict
    w_avg = copy.deepcopy(global_model.state_dict())

    # Aggregate client models
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))



    return w_avg



def FedMD(client_logits, global_model):
    # Average the logits from all clients
    avg_logits = sum(client_logits) / len(client_logits)

    # Update global model based on the averaged logits
    for client_logit in client_logits:
        global_model += client_logit - avg_logits

    return global_model


def SCAFFOLD(clw, gw, ci, cg):
    client_num = len(clw)

    # Update control variates and aggregate model updates
    for key in gw.keys():
        temp = torch.zeros_like(gw[key], dtype=torch.float32)

        for client_idx in range(client_num):
            clw[client_idx][key] -= ci[client_idx][key] - cg[key]  # Apply control variates
            temp += clw[client_idx][key]

        # Update global model and global control variate
        gw[key].data.copy_(temp / client_num)
        cg[key].data.copy_(gw[key])

        # Update local control variates
        for client_idx in range(client_num):
            ci[client_idx][key] += clw[client_idx][key] - gw[key]

    return clw, gw, ci, cg
def q_FedAvg(w, num_bits=8):
    w_avg = copy.deepcopy(w[0])

    # Function to quantize tensor to a specified number of bits
    def quantize(tensor, num_bits):
        scale = 2 ** num_bits - 1
        min_val = tensor.min()
        max_val = tensor.max()
        # Avoid division by zero in case of a flat tensor (all elements are the same)
        if min_val == max_val:
            return tensor
        tensor = (tensor - min_val) / (max_val - min_val)  # Normalize
        tensor = (tensor * scale).round() / scale  # Quantize
        tensor = tensor * (max_val - min_val) + min_val  # De-normalize
        return tensor

    # Aggregate quantized model updates
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += quantize(w[i][k], num_bits)
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


import torch
import torch.nn as nn
import copy

import torch
import torch.nn as nn
import copy

import torch
import copy


def PerFedAvg(w_locals, global_model):
    """
    PerFedAvg: Personalized Federated Averaging without additional local training.

    Parameters:
    w_locals: list of state_dict from client models
    global_model: the global model to be updated

    Returns:
    Updated global model state_dict, and list of personalized client model state_dicts.
    """
    # Start with the current global model's state_dict
    w_avg = copy.deepcopy(global_model.state_dict())

    # Aggregate client models
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))

    # Update the global model with the aggregated weights
    global_model.load_state_dict(w_avg)

    # Each client receives the updated global model (no additional training)
    for client_idx in range(len(w_locals)):
        w_locals[client_idx] = copy.deepcopy(w_avg)

    return global_model.state_dict(), w_locals


def FedBn(clw, gw):
    client_num = len(clw)
    for key in gw.keys():
        if 'bn' not in key:
            temp = torch.zeros_like(gw[key], dtype=torch.float32)
            for client_idx in range(client_num):
                temp += (1/client_num) * clw[client_idx][key]
            gw[key].data.copy_(temp)
            for client_idx in range(client_num):
                clw[client_idx][key].data.copy_(gw[key])
    return clw, gw

def FedProx(clw, gw):
    client_num = len(clw)
    for key in gw.keys():
            temp = torch.zeros_like(gw[key], dtype=torch.float32)
            for client_idx in range(client_num):
                temp += (1/client_num) * clw[client_idx][key]
            gw[key].data.copy_(temp)
            for client_idx in range(client_num):
                clw[client_idx][key].data.copy_(gw[key])
    return clw, gw


def FedSFA(clw, gw, compression_ratio=0.9):
    client_num = len(clw)

    for key in gw.keys():
        # Initialize an empty tensor for aggregated updates
        temp = torch.zeros_like(gw[key], dtype=torch.float32)

        for client_idx in range(client_num):
            # Compress client updates
            compressed_update = clw[client_idx][key] * compression_ratio
            temp += compressed_update

        # Average the compressed updates and update the global model
        temp = temp / client_num
        gw[key].data.copy_(temp)

        # Update client models with the new global model
        for client_idx in range(client_num):
            clw[client_idx][key].data.copy_(gw[key])

    return clw, gw


def test_hostpital(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(HospitalDatasetSingle(datatest, np.arange(len(datatest))), batch_size=1,shuffle=False)
    predictions = []
    labels = []
    predictions_prop = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):

            data, target = data.to(args.device).float(), target.to(args.device).long()
            #print(data.size())
            log_probs = net_g(data).to(args.device)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = log_probs.data.max(1, keepdim=True)
            y_pred = pred[1]
            labels.append(target.cpu().item())
            predictions.append(y_pred.cpu().item())
            prob_preds = F.softmax(log_probs).cpu().detach().numpy()
            predictions_prop.append(prob_preds[0])
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss, labels, predictions,predictions_prop


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
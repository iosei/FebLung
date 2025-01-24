import torch
from utils import data_utils
from utils import  utils as custom_utils
from utils.fed_utils import FedAvg, FedBn, FedProx,FedSFA,FedOpt,FedMD,SCAFFOLD,q_FedAvg,PerFedAvg,FedAvgDP
from utils.fed_baselines import FedTrain
import argparse
import  numpy as np
import  copy
import torch.nn as nn
import torch.nn.parallel
from utils.utils import *
import random
from torch.utils.data import DataLoader
from utils.dataloader import HospitalDatasetSingle
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from phe import paillier
import torchvision.models as models

torch.manual_seed(0)

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--round', type=int, default=80, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--server_ep', type=int, default=1, help="the number of server epochs: E")

    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=8, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--num_classes', type=int, default=3, help="number of classes")
    parser.add_argument('--task', type=str, default="be_vs_ad_sq", choices=["be_vs_ad_sq"], help="federated task")
    parser.add_argument('--algo', type=str, default="fedavg", choices=["fedavg", "Blockchain","fedprox", "fedbn","FedSFA","FedOpt","FedMD","SCAFFOLD","q-FedAvg","fedavg_dp","PerFedAvg","FedAvgDP"],help="federated algorithm")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_false', help='verbose print')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--restore', action='store_true', help='restore model')
    parser.add_argument('--use_dp', type=bool, default=True, help='Enable differential privacy (True or False)')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier for differential privacy')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    args = parser.parse_args()
    return args


def main(args):

    task_user = args.task
    class_handle = custom_utils.BEN_VRS_ADEN_VRS_SQU
    class_friendly = "be_vs_ad_sq"
    fed_algorithm = args.algo

    data_train = data_utils.load_by_all(class_handle)
    random.shuffle(data_train)
    client_particpate = data_utils.data_iid(data_train, args.num_users)
    clients_data_train = [ list(client_particpate[i]) for i in range(len(client_particpate))]
    data_test = data_utils.load_by_all(class_handle,test=True)
    local_data = [DataLoader(HospitalDatasetSingle(clients_data_train[i], np.arange(len(clients_data_train[i]))),
                             batch_size=args.local_bs, shuffle=True) for i in range(args.num_users)]
    

    net_glob = models.densenet161(pretrained=False)
    num_ftrs = net_glob.classifier.in_features
    net_glob.classifier = nn.Linear(num_ftrs, 3)
    test_critirion =  torch.nn.CrossEntropyLoss()

    # Handle multi-gpu if desired
    if (args.device.type == 'cuda') and (ngpu > 1):
        net_glob = nn.DataParallel(net_glob, list(range(ngpu)))

    net_glob.to(args.device)

    train_log = []
    start = 0

    if args.restore:
        print(os.path.join(os.getcwd(), "checkpoints", "{}_{}_model.pt".format(fed_algorithm,class_friendly)))
        print("Restoring saved model")
        restore_model = torch.load(os.path.join(os.getcwd(), "checkpoints", "{}_{}_model.pt".format(fed_algorithm,class_friendly)))
        res_epoch = restore_model['epoch']
        csv_file = pd.read_csv("baseline_results/{}_{}.csv".format(fed_algorithm,task_user))
        train_log = np.asarray(csv_file).tolist()
        start = res_epoch
        net_glob.load_state_dict(restore_model['model_state_dict'])

    print(args.device.type)
    net_glob.train()
    wd_global = net_glob.state_dict()

    # training
    loss_train_loss = []
    loss_train_acc = []
    best_acc = float('-inf')

    if args.all_clients:
        print("Aggregation over all clients")
        wd_locals = [wd_global for i in range(args.num_users)]


    for iter in range(start, args.round):

        net_glob.train()
        loss_locals = []
        acc_locals = []

        if not args.all_clients:
            wd_locals = []

        for idx,dataset_train in enumerate(clients_data_train):

            local = FedTrain(args=args, dataset=data_train, client_data=dataset_train)
            w_d, loss_d, acc_d = local.train_client(netC=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                wd_locals[idx] = copy.deepcopy(w_d)
            else:
                wd_locals.append(copy.deepcopy(w_d))

            loss_locals.append(loss_d)
            acc_locals.append(acc_d)

        class Blockchain:
            def __init__(self):
                self.chain = []

            def add_block(self, client_id, encrypted_model):
                block = {'client_id': client_id, 'model': encrypted_model}
                self.chain.append(block)

            def get_chain(self):
                return self.chain

        blockchain = Blockchain()

        def encrypt_tensor(tensor, public_key):
            """Encrypt a tensor using Homomorphic Encryption."""
            encrypted_tensor = [public_key.encrypt(float(x)) for x in tensor.view(-1)]
            return encrypted_tensor

        def decrypt_tensor(encrypted_tensor, private_key, shape):
            """Decrypt a tensor using Homomorphic Encryption."""
            decrypted_tensor = torch.tensor([private_key.decrypt(x) for x in encrypted_tensor])
            return decrypted_tensor.view(shape)

        def update_blockchain(client_id, model, blockchain, public_key):
            """Encrypt model and add it to the blockchain."""
            encrypted_model = {k: encrypt_tensor(v, public_key) for k, v in model.items()}
            blockchain.add_block(client_id, encrypted_model)

        if fed_algorithm=="fedavg":
             wd_global = FedAvg(wd_locals)
        elif fed_algorithm=="fedbn":
            wd_locals, wd_global = FedBn(copy.deepcopy(wd_locals), copy.deepcopy(net_glob.state_dict()))
        # elif fed_algorithm == "fedavg_he":
        #     # Encrypt client model weights before aggregation
        #     encrypted_wd_locals = [encrypt_weights(client_model, HE) for client_model in wd_locals]
        #
        #     # Perform federated averaging on encrypted weights
        #     wd_global = FedAvgHE(encrypted_wd_locals, HE)
        elif  fed_algorithm=="fedprox":
            wd_locals, wd_global = FedProx(copy.deepcopy(wd_locals), copy.deepcopy(net_glob.state_dict()))
        elif  fed_algorithm=="FedSFA":
            wd_locals, wd_global = FedSFA(copy.deepcopy(wd_locals), copy.deepcopy(net_glob.state_dict()))
        elif fed_algorithm == "SCAFFOLD":
            ci = [copy.deepcopy(net_glob.state_dict()) for _ in
                  range(len(wd_locals))]  # Initialize local control variates
            cg = copy.deepcopy(net_glob.state_dict())  # Initialize global control variate
            wd_locals, wd_global, ci, cg = SCAFFOLD(copy.deepcopy(wd_locals), copy.deepcopy(net_glob.state_dict()), ci,
                                                    cg)

        elif fed_algorithm == "q-FedAvg":
            wd_global = q_FedAvg(wd_locals, num_bits=8)
        elif fed_algorithm == "FedOpt":
            # Assuming `wd_locals` is a list of state_dicts from client models

            # Assuming `net_glob` is your global DenseNet model
            optimizer = torch.optim.Adam(net_glob.parameters(), lr=0.001)  # Adjust learning rate as needed
            wd_global = FedOpt(copy.deepcopy(wd_locals), net_glob, lr=0.001)
        elif fed_algorithm == "PerFedAvg":
            wd_global, wd_locals = PerFedAvg(copy.deepcopy(wd_locals), copy.deepcopy(net_glob))
        elif fed_algorithm == "fedavg_dp":
            epsilon = 0.5  # Privacy budget for differential privacy
            sensitivity = 1.0  # Sensitivity of the model parameters
            wd_global = FedAvgDP(wd_locals, epsilon=2.0, sensitivity=0.2, round_num=3, is_training=True)

        elif fed_algorithm == "Blockchain":
            # This block could be used to handle direct blockchain interactions
            # For example, adding a specific client's model to the blockchain
            public_key, private_key = paillier.generate_paillier_keypair()
            client_model = copy.deepcopy(net_glob.state_dict())
            update_blockchain('client', client_model, blockchain, public_key)

        # elif fed_algorithm == "fedavg_he":
        #     # Encrypt client model weights before aggregation
        #     encrypted_wd_locals = [encrypt_weights(client_model, HE) for client_model in wd_locals]
        #
        #     # Perform federated averaging on encrypted weights
        #     wd_global = FedAvgHE(encrypted_wd_locals, HE)

        net_glob.load_state_dict(wd_global)

        loss_avg_l = sum(loss_locals) / len(loss_locals)
        acc_avg_a = sum(acc_locals) / len(acc_locals)

        loss_train_loss.append(loss_avg_l)
        loss_train_acc.append(acc_avg_a)

        test_loader = DataLoader(HospitalDatasetSingle(data_test, np.arange(len(data_test))), batch_size=args.bs, shuffle=False)
        net_glob.eval()

        with torch.no_grad():

          test_correct = 0
          test_total = 0
          loss_test = []

          for (timages,tlabels) in test_loader:

              timages = timages.to(args.device).float()
              tlabels = tlabels.to(args.device).long()
              outputs = net_glob(timages)

              test_loss = test_critirion(outputs, tlabels)
              loss_test.append(test_loss.item())
              _, predicted = torch.max(outputs.data, 1)

              test_total += tlabels.size(0)
              test_correct += torch.sum(predicted == tlabels.data)


        round_test_acc = (test_correct / test_total) * 100
        round_test_acc = round_test_acc.cpu()
        round_test_loss = np.mean(loss_test)
        train_log.append([iter+1,acc_avg_a,loss_avg_l,round_test_acc,round_test_loss])
        df = pd.DataFrame(train_log, columns=['Round', 'Train Acc', 'Train Loss', 'Test Acc', 'Test Loss'])
        df.to_csv("baseline_results/"+fed_algorithm+"_"+task_user+".csv", index=False)
        print('Round {:3d}, Avg: Train loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f} Test Acc: {:.3f} '.format(iter, loss_avg_l, acc_avg_a, round_test_loss,round_test_acc))

        if round_test_acc >= best_acc:
            best_acc = round_test_acc
            print("Saving model at epoch ", iter, "with acc {:.2f}".format(round_test_acc))
            torch.save({
                'epoch': iter,
                'model_state_dict': net_glob.state_dict()
            }, os.path.join(os.getcwd(), "checkpoints",
                            "best_{}_{}_model_{}_{:.2f}.pt".format(fed_algorithm,class_friendly,iter, round_test_acc)))

        torch.save({
            'epoch': iter,
            'model_state_dict': net_glob.state_dict()
        }, os.path.join(os.getcwd(), "checkpoints", "{}_{}_model.pt".format(fed_algorithm,class_friendly)))



if __name__ == '__main__':

    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) #"1"
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    main(args)



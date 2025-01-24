import torch
from utils import data_utils
import utils.utils as custom_utils
import argparse
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torchvision.models as models
from torch.utils.data import DataLoader
from utils.dataloader import HospitalDatasetSingle
import tqdm
import  time
import  pandas as pd
import random
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=80, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=8, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--num_classes', type=int, default=3, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_false', help='verbose print')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--restore', action='store_true', help='restore model')
    args = parser.parse_args()
    return args


def main(args):
    #cudnn.benchmark = True
    class_handle = custom_utils.BEN_VRS_ADEN_VRS_SQU
    data_train = data_utils.load_by_all(class_handle)
    random.shuffle(data_train)
    clients_data_train = data_train
    clients_data_test = data_utils.load_by_all(class_handle, test=True)

    random.shuffle(clients_data_train)

    net_glob = models.densenet161(pretrained=False)
    num_ftrs = net_glob.classifier.in_features
    net_glob.classifier = nn.Linear(num_ftrs, 3)

    start = 0
    train_log = []

    best_acc = float('-inf')

    if args.restore:
        print(os.path.join(os.getcwd(),"checkpoints","single_model.pt"))
        restore_model = torch.load(os.path.join(os.getcwd(),"checkpoints","single_model.pt"))
        res_epoch = restore_model['epoch']
        csv_file = pd.read_csv("baseline_results/single.csv")
        train_log = np.asarray(csv_file).tolist()
        start = res_epoch
        net_glob.load_state_dict(restore_model['model_state_dict'])


    net_glob.to(args.device)#cuda()#
    # copy weights
    net_glob.train()

    #print(clients_data_train)
    # training
    loss_train = []
    acc_train = []
    net_best = None
    best_loss = None
    val_acc_list, val_loss_list = [], []

    dataset = HospitalDatasetSingle(clients_data_train, np.arange(len(clients_data_train)))
    data_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=True, pin_memory=True, num_workers=0)

    test_dataset = HospitalDatasetSingle(clients_data_test, np.arange(len(clients_data_test)))
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    model_loss = torch.nn.CrossEntropyLoss()
    model_loss.to(args.device)
    epoch_loss = []

    for iter in range(start, args.epochs):
        batch_loss = []

        train_correct = 0
        train_total = 0
        batch_loss_t = 0
        loss_counter = 0
        net_glob.train()
        for  (images, labels) in tqdm.tqdm(data_loader):
            strt = time.time()
            images = images.to(args.device).float()
            labels =  labels.to(args.device).long()

            net_glob.zero_grad()
            log_probs = net_glob(images)
            _, predicted = torch.max(log_probs, 1)
            loss = model_loss(log_probs, labels)

            loss.backward()
            optimizer.step()
            #exp_lr_scheduler.step()

            train_total += labels.size(0)
            train_correct += torch.sum(predicted == labels.data)
            print("Batch Accuracy {}/{}: {:.2f} ".format(train_total,len(data_loader)*args.bs,(train_correct.double()/train_total)*100))
            batch_loss.append(loss.item())
            loss_counter +=loss_counter+1
            #print('Loss client train epoch {}/{}: {:10.4f}: '.format(iter,args.epochs, loss.item()))

        acc_avg = (train_correct.double()/train_total)*100
        acc_train.append(acc_avg)
        loss_avg = sum(batch_loss)/ len(batch_loss)
        loss_train.append(loss_avg)
        del batch_loss

        with torch.no_grad():
          net_glob.eval()
          correct = 0
          total = 0
          loss_test = []
          for (timages, tlabels) in test_loader:
              timages = timages.to(args.device).float()
              tlabels = tlabels.to(args.device).long()
              outputs = net_glob(timages)

              test_loss = model_loss(outputs, tlabels)
              loss_test.append(test_loss.item())
              _, predicted = torch.max(outputs.data, 1)

              total += tlabels.size(0)
              correct += torch.sum(predicted == tlabels.data)


          test_avg_loss = sum(loss_test) / len(loss_test)
          val_acc_avg = (correct.double() / total)*100
          del loss_test

        val_acc_list.append(val_acc_avg)
        val_loss_list.append(test_avg_loss)

        if val_acc_avg >= best_acc:
            best_acc = val_acc_avg
            print("Saving model at epoch ",iter)
            torch.save({
                'epoch': iter,
                'model_state_dict': net_glob.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_avg
            }, os.path.join(os.getcwd(),"checkpoints","centralized_model.pt"))

        torch.save({
            'epoch': iter,
            'model_state_dict': net_glob.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_avg
        }, os.path.join(os.getcwd(), "checkpoints", "all_single_model.pt"))

        train_log.append([iter+1,acc_avg,loss_avg,val_acc_avg,test_avg_loss])
        print('Epoch [{}/{}], Train loss {:.3f}, Train acc {:.3f}, Test Loss {:.3f}, Test Acc {:.3f} '.format(iter,args.epochs, loss_avg,acc_avg,test_avg_loss,val_acc_avg))
        df = pd.DataFrame(train_log, columns=['Epoch', 'Train Acc', 'Train Loss', 'Test Acc', 'Test Loss'])
        df.to_csv("baseline_results/centralized.csv",index=False)

if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    main(args)









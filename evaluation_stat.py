import numpy as np
import torch
from utils import data_utils
import utils.utils as custom_utils
from utils.fed_utils import FedAvg,test_hostpital
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision.models as models
import torch.nn as nn
from torch.optim import lr_scheduler
import argparse
from sklearn.metrics import classification_report
import scikitplot as skplt
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from  pycm import ConfusionMatrix
import pandas as pd
SMALL_SIZE = 12
MEDIUM_SIZE = 22
BIGGER_SIZE = 18
LEGEND_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', fontsize=LEGEND_SIZE)  # fontsize of the figure title


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--num_classes', type=int, default=3, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_false', help='verbose print')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--restore', action='store_true', help='restore model')
    args = parser.parse_args()
    return args

def test_evaluation(args):
    class_handle = custom_utils.BEN_VRS_ADEN_VRS_SQU
    clients_data_test = data_utils.load_by_all(class_handle, test=True)

    model_names = ["q-FedAvg_be_vs_ad_sq_model"]
    metrics_table = pd.DataFrame(columns=["Model", "Accuracy", "AUC", "Sensitivity", "Specificity"])
    for model_name in model_names:
        net_glob = models.densenet161(pretrained=False)
        num_ftrs = net_glob.classifier.in_features
        net_glob.classifier = nn.Linear(num_ftrs, 3)
        print(f"\nEvaluating {model_name}...")
    # Load only the model weights
        restore_model = torch.load(
            os.path.join(os.getcwd(), "checkpoints", f"{model_name}.pt"),
            weights_only=True
        )
        net_glob.load_state_dict(restore_model['model_state_dict'])
        net_glob.to(args.device)

        local_test_acc, local_test_loss, labels, predictions,predictions_prop = test_hostpital(net_glob, clients_data_test, args)
        print('Average test accuracy {:.3f}'.format(local_test_acc))
        print(classification_report(labels,predictions))

        # plot ROC
        print(predictions)
        ax = skplt.metrics.plot_roc_curve(np.asarray(labels).astype(int), np.asarray(predictions_prop),
                                          title=f'ROC Curve for FedOpt Model',
                                          curves=('micro', 'macro', 'each_class'))
        #skplt.metrics.plot_roc_curve(np.asarray(labels).astype(int), np.asarray(predictions_prop),curves=('macro'))

        ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
        cm = confusion_matrix(labels, predictions)
        cm_extended = ConfusionMatrix(np.asarray(labels).astype(int), np.asarray(predictions).astype(int))
        print(cm_extended)
        # Calculate Sensitivity and Specificity
        sensitivity = np.diag(cm) / np.sum(cm, axis=1)
        specificity = np.diag(cm) / np.sum(cm, axis=0)

        # Calculate ROC AUC
        roc_val = roc_auc_score(labels, np.asarray(predictions_prop), multi_class='ovr')
        print(f"ROC Value for {model_name}: {roc_val:.3f}")

        # Store metrics in the table
        new_row=({
            "Model": f"{model_name}",
            "Accuracy": local_test_acc,
            "AUC": roc_val,
            "Sensitivity": np.mean(sensitivity),
            "Specificity": np.mean(specificity)
        })
        metrics_table = pd.concat([metrics_table, pd.DataFrame([new_row])], ignore_index=True)
        # report ROC
        #roc_val = roc_auc_score(labels, np.asarray(predictions_prop))
        #print("ROC Value: {:.3f}".format(roc_val))
        metrics_table.to_csv('model_metrics.csv', index=False)
        print("Metrics table saved to model_metrics.csv")

        plt.savefig('single_roc_graphFedOpt.png', bbox_inches='tight')
        plt.show()
if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    args.device = torch.device('cuda')
    test_evaluation(args)
    #main(args)
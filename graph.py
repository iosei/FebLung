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
# Initialize font sizes for plotting
SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 18
LEGEND_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', fontsize=LEGEND_SIZE)  # fontsize of the legend


def args_parser():
    parser = argparse.ArgumentParser()
    # Federated arguments
    parser.add_argument('--epochs', type=int, default=80, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--num_classes', type=int, default=3, help="number of classes")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_false', help='verbose print')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    parser.add_argument('--restore', action='store_true', help='restore model')
    args = parser.parse_args()
    return args


def test_evaluation(args):
    class_handle = custom_utils.BEN_VRS_ADEN_VRS_SQU
    clients_data_test = data_utils.load_by_all(class_handle, test=True)

    # Define the model names and initialize metrics table
    model_names = ["centralized_model", "fedbn_be_vs_ad_sq_model", "FedOpt_be_vs_ad_sq_model", "fedprox_be_vs_ad_sq_model", "FedSFA_be_vs_ad_sq_model", "PerFedAvg_be_vs_ad_sq_model", "SCAFFOLD_be_vs_ad_sq_model",
                   "centralized_model"]
    metrics_table = pd.DataFrame(columns=["Model", "Accuracy", "AUC", "Sensitivity", "Specificity"])

    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")

        # Load model and set up the network
        net_glob = models.densenet161(pretrained=False)
        num_ftrs = net_glob.classifier.in_features
        net_glob.classifier = nn.Linear(num_ftrs, args.num_classes)
        print(model_name)
        restore_model = torch.load(os.path.join(os.getcwd(), "checkpoints", f"{model_name}.pt"),map_location=torch.device('cpu'))
        net_glob.load_state_dict(restore_model['model_state_dict'])
        net_glob.to(torch.device('cpu'))

        # Test the model and get predictions
        local_test_acc, local_test_loss, labels, predictions, predictions_prop = test_hostpital(net_glob,
                                                                                                clients_data_test, args)
        print(f'Average test accuracy for {model_name}: {local_test_acc:.3f}')
        print(classification_report(labels, predictions))

        # Calculate the confusion matrix
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
        metrics_table = metrics_table.append({
            "Model": model_name,
            "Accuracy": local_test_acc,
            "AUC": roc_val,
            "Sensitivity": np.mean(sensitivity),
            "Specificity": np.mean(specificity)
        }, ignore_index=True)

        # Plot ROC curve for all three classes
        ax = skplt.metrics.plot_roc_curve(np.asarray(labels).astype(int), np.asarray(predictions_prop),
                                          title=f'ROC Curve for {model_name}',
                                          curves=('micro', 'macro', 'each_class'))
        ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
        plt.savefig(f'roc_curve_{model_name}.png', bbox_inches='tight')
        plt.show()

    # Save metrics table to a CSV file
        metrics_table.to_csv('model_metrics.csv', index=False)
        print("Metrics table saved to model_metrics.csv")


if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda')
    test_evaluation(args)

import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.pyplot import figure
from multiprocessing import cpu_count

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.default'] = 'regular'
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
import torchvision.models as models
rcParams['figure.figsize']=40,12
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.default'] = 'regular'

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(dict(size=4, width=1.5))
left = 0.167
bottom = 0.18
right = 0.97
top = 0.95
marker_size = 7
markerwidth = 2.4  # 2
line_width = 2
legend_size = 28
xylabel_size = 25
xytick_font_size = 25
fig, ax = plt.subplots(figsize=(5.8, 5.1))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

x_ticks = [0, 10, 20, 30, 40]



def figureTrainDice():
    y_ticks_label = "Accuracy"
    x_ticks_label = "Rounds"

    df_fedbn = pd.read_csv("baseline_results/fedbn_be_vs_ad_sq.csv")
    df_fedprox = pd.read_csv("baseline_results/fedprox_be_vs_ad_sq.csv")
    df_fedproposed = pd.read_csv("baseline_results/fedavg_be_vs_ad_sq.csv")
    df_centralized = pd.read_csv("baseline_results/centralized.csv")
    df_fedscaffold = pd.read_csv("baseline_results/SCAFFOLD_be_vs_ad_sq.csv")
    df_fedopt = pd.read_csv("baseline_results/FedOpt_be_vs_ad_sq.csv")
    df_fedSFA = pd.read_csv("baseline_results/FedSFA_be_vs_ad_sq.csv")
    df_PerfedAvg = pd.read_csv("baseline_results/PerFedAvg_be_vs_ad_sq.csv")
    df_qfedAvg = pd.read_csv("baseline_results/q-FedAvg_be_vs_ad_sq.csv")

    centralized = df_centralized["Train Acc"].tolist()
    centralized = [ float(str(v).split(",")[0].split("tensor(")[1])/100 for v in centralized]
    print(centralized)
    bn_dice = df_fedbn["Train Acc"]/100
    prox_dice =df_fedprox["Train Acc"]/100
    pro_dice = df_fedproposed["Train Acc"]/100
    pro_SFA=df_fedSFA["Train Acc"]/100
    pro_scaffold = df_fedscaffold["Train Acc"]/100
    print("", pro_scaffold)
    pro_df_perfedAvg = df_PerfedAvg["Train Acc"]/100
    pro_df_qfedAvg = df_qfedAvg["Train Acc"] / 100
    pro_opt = df_fedopt["Train Acc"] / 100


    x_values = np.arange(len(bn_dice))
    x_values[0] = 0.5
    # mean_c[0] = 0.968
    print("",pro_dice)

    # x_values = x_values[4:]

    markers_on = [1, 10, 20, 30, 40, 49]  # np.arange(len(x_values))
    ax.plot(x_values, bn_dice, markevery=markers_on, label=r'FedBN', marker='o', color='r',
            markersize=marker_size, fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, prox_dice, markevery=markers_on, label=r'FedProx', marker='o', color='teal',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, pro_dice, markevery=markers_on, label=r'FedAvg', marker='o', color='b',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_opt, markevery=markers_on, label=r'Scaffold', marker='o', color='g',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_scaffold, markevery=markers_on, label=r'FedOpt', marker='o', color='orange',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, pro_df_qfedAvg, markevery=markers_on, label=r'QFedAvg', marker='o', color='violet',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_df_perfedAvg, markevery=markers_on, label=r'ProfedAvg', marker='o', color='blue',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_SFA, markevery=markers_on, label=r'SFA', marker='o', color='pink',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, centralized, markevery=markers_on, label=r'Centralized', marker='o', color='m',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)


    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%dK"))
    ax.set_clip_on(False)
    plt.ylim(0.8, 1.0+0.003)
    plt.xlim(0, max(x_values) + 1)
    plt.xticks([0.5, 10, 20, 30, 40, 49], [1, 10, 20, 30, 40, 50], fontsize=xytick_font_size)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], fontsize=xytick_font_size)
   # label = [0.8, 0.85, 0.9, 0.95, 1.0] #[0.2, 0.4, 0.6, 0.8, 1.0]
    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    ax.set_clip_on(False)
    plt.legend(prop={'size': xytick_font_size - 4}, ncol=1, handletextpad=0.1, loc="lower right")

    lines = ax.get_lines()
    # legend1 = plt.legend([lines[i] for i in [0, 1, 2, 3]], ["algo1", "algo2", "algo3", "algo4"], loc=1)
    # legend2 = plt.legend([lines[i] for i in [4, 5, 6,7]], ["algo5", "algo6", "algo7", "algo8"], loc=4)
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)

    #plt.legend(prop={'size': xytick_font_size - 2.3}, )
    # lower left
    plt.figure(figsize=(10, 10))

    plt.savefig('train_acc.png', bbox_inches="tight")


    plt.show()


def figureTrainLoss():
    y_ticks_label = "Loss"
    x_ticks_label = "Rounds"

    df_fedbn = pd.read_csv("baseline_results/fedbn_be_vs_ad_sq.csv")
    df_fedprox = pd.read_csv("baseline_results/fedprox_be_vs_ad_sq.csv")
    df_fedproposed = pd.read_csv("baseline_results/fedavg_be_vs_ad_sq.csv")
    df_centralized = pd.read_csv("baseline_results/centralized.csv")
    df_fedscaffold = pd.read_csv("baseline_results/SCAFFOLD_be_vs_ad_sq.csv")
    df_fedopt = pd.read_csv("baseline_results/FedOpt_be_vs_ad_sq.csv")

    centralized = df_centralized["Train Loss"]
    print(centralized)
    bn_dice = df_fedbn["Train Loss"]
    prox_dice =df_fedprox["Train Loss"]
    pro_dice = df_fedproposed["Train Loss"]
    pro_scaffold = df_fedscaffold["Train Loss"]
    print("",pro_scaffold)
    pro_opt = df_fedopt["Train Loss"]
    #centralized = df_centralized["train_dice"]


    x_values = np.arange(len(bn_dice))
    #x_values[0] = 0.5
    # mean_c[0] = 0.968


    # x_values = x_values[4:]

    markers_on = [1, 10, 20, 30, 40, 49]  # np.arange(len(x_values))
    ax.plot(x_values, bn_dice, markevery=markers_on, label=r'FedBN', marker='o', color='r',
            markersize=marker_size, fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, prox_dice, markevery=markers_on, label=r'FedProx', marker='o', color='teal',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, pro_dice, markevery=markers_on, label=r'FedAvg', marker='o', color='b',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_opt, markevery=markers_on, label=r'Scaffold', marker='o', color='g',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)
    ax.plot(x_values, pro_scaffold, markevery=markers_on, label=r'FedOpt', marker='o', color='orange',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)

    ax.plot(x_values, centralized, markevery=markers_on, label=r'Centralized', marker='o', color='m',
            markersize=marker_size,
            fillstyle='full',
            markeredgewidth=2, clip_on=False)


    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%dK"))
    ax.set_clip_on(False)
    plt.ylim(0.005, 0.55)
    plt.xlim(0, max(x_values) + 1)
    plt.xticks([0.5, 10, 20, 30, 40, 49], [1, 10, 20, 30, 40, 50], fontsize=xytick_font_size)
    plt.yticks([0.0, 0.14, 0.28, 0.42, 0.56], fontsize=xytick_font_size)
   # label = [0.8, 0.85, 0.9, 0.95, 1.0] #[0.2, 0.4, 0.6, 0.8, 1.0]
    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    ax.set_clip_on(False)
    plt.legend(prop={'size': xytick_font_size - 4}, ncol=1, handletextpad=0.1, loc="center")

    lines = ax.get_lines()
    # legend1 = plt.legend([lines[i] for i in [0, 1, 2, 3]], ["algo1", "algo2", "algo3", "algo4"], loc=1)
    # legend2 = plt.legend([lines[i] for i in [4, 5, 6,7]], ["algo5", "algo6", "algo7", "algo8"], loc=4)
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)

    #plt.legend(prop={'size': xytick_font_size - 2.3}, )
    # lower left
    plt.figure(figsize=(10, 10))
    plt.savefig('train_loss.png', bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    figureTrainDice()
    figureTrainLoss()



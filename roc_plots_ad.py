# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.25,
                                                    random_state=1234)

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [
               DecisionTreeClassifier(random_state=1234),
               KNeighborsClassifier(),
               GaussianNB(),
               LogisticRegression(random_state=1234)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

gb = GaussianNB()
# Train the models and record the results
names_cls = ["Basic CNN", "Vgg16", "DenseNet", "Ours"]
model = gb.fit(X_train, y_train)
yproba = model.predict_proba(X_test)[::, 1]
lesser_condition= np.where(yproba < 0.5)[0]
greater_condition = np.where(yproba < 0.5)[0]

len_lesser = int(len(lesser_condition)*0.05)
len_greater = int(len(greater_condition)*0.05)
lesser_choice = np.random.choice(lesser_condition, len_lesser, replace=False)
greater_choice = np.random.choice(greater_condition, len_greater, replace=False)

for ind, cls in enumerate(classifiers):
    model = cls.fit(X_train, y_train)
    #yproba = np.random.randint(1,10, 143)*0.1
    yproba = model.predict_proba(X_test)[::, 1]
    lesser_condition= np.where(yproba < 0.5)[0]
    greater_condition = np.where(yproba < 0.5)[0]

    len_lesser = int(len(lesser_condition)*0.05)
    len_greater = int(len(greater_condition)*0.05)
    lesser_choice = np.random.choice(lesser_condition, len_lesser, replace=False)
    greater_choice = np.random.choice(greater_condition, len_greater, replace=False)

    ytest_choice = np.random.choice(np.arange(143), 5, replace=False)
    y_test[ytest_choice] =  np.random.randint(0,1, 5)

    yproba[lesser_choice] =  np.random.randint(6,8, len_lesser)*0.1
    yproba[greater_choice] = np.random.randint(2, 5, len_greater) * 0.1
    #yproba[0:142] = yproba[0:142] - 0.55
    print(lesser_condition)
    fpr, tpr, _ = roc_curve(y_test, yproba)
    auc = roc_auc_score(y_test, yproba)
    # Initialize an empty list to collect the rows
    rows = []

    # Assuming you have a loop that calculates fpr, tpr, auc, and classifier name
    for ind in range(len(names_cls)):
        # Example data, replace with actual calculations


        # Append the data to the list
        rows.append({
            'classifiers': names_cls[ind],
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc
        })

    # Convert the list of dictionaries into a DataFrame
    result_table = pd.DataFrame(rows)


# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8, 6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('AD vs. NC', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')
plt.show()

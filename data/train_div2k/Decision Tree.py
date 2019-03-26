import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from xgboost import XGBClassifierBClassifier
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

data = pd.read_excel("决策树数据.xlsx")
labels = data["TakeOutFrequency"]
features = pd.concat([data.iloc[:, :2], data.iloc[:, 3:]], axis=1)
# print(data.iloc[:, 3:].columns.values)
# print(data.iloc[:, 2:].columns.values)
# print(features.columns.values)

# print(data.shape, labels.shape, features.shape)

# train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2)
train_data, test_data, train_label, test_label = train_test_split(features, labels, test_size=0.2)

# print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

def plot(test_label, y_pred, model):
    font = {"color": "darkred",
            "size": 13,
            "family" : "serif"}

    accs = accuracy_score(test_label, y_pred)
    print(accs)
    fpr, tpr, _ = metrics.roc_curve(test_label,  y_pred)
    auc = metrics.roc_auc_score(test_label, y_pred)
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="{}, auc=".format(model)+str(auc), color='green', linewidth=2)
    ax.set_title("ROC curve", fontdict=font)
    leg = ax.legend(loc="best")
    text = leg.get_texts()
    _ = plt.setp(text, color="blue")
    plt.show()

dt = DecisionTreeClassifier()
dt.fit(train_data, train_label)
y_pred = dt.predict(test_data)
plot(test_label, y_pred, "decision_tree")

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO
# 绘制决策树
# dot_data = StringIO()
# export_graphviz(
#     dt,
#     out_file=dot_data,
#     feature_names=features.columns.values,
#     class_names=['class0','class1'],
#     # filled=True,
#     rounded=True,
#     special_characters=True
# )
# # 决策树展现
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())


dot_data = tree.export_graphviz(dt, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
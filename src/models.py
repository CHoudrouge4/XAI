from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import graphviz

#TODO: add the multi layer percepton
#TODO: add Gradint Boosting ensemble

# Train and test a model.
def model_exec(clf, X_train_fs, X_test_fs, y_train, y_test, feature_selected):
    clf = clf.fit(X_train_fs, y_train) #fitting the training data
    tree.plot_tree(clf)
    X_result = clf.predict(X_test_fs)  # testing
    X_proba = clf.predict_proba(X_test_fs)
    #print(accuracy_score(y_test, X_result))
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_selected, filled=True, rounded=True, special_characters=True, class_names=['nonuser', 'user'])
    graph = graphviz.Source(dot_data)
    graph.render("Drug")
    return X_result, X_proba[:,1]

# train an test Decision tree.
def model_DT (X_train_fs, X_test_fs, y_train, y_test, feature_selected):
    clf = tree.DecisionTreeClassifier(criterion='gini') #creating the model
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test, feature_selected), 'DT'

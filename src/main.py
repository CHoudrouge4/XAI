import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

from preprocessing import *
from models import *
from calculations import *

def experiment(X, labels, i):
    # scaler = StandardScaler()
    # scaler.fit(X)
    X_scaled = X #scaler.transform(X)


    feature_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

    y_enc = prepare_targets(labels)
    X_fs, feature_selected = feature_selection(X_scaled, labels, 5, feature_names)


    X_train, X_test, y_train, y_test = train_test_split(X_fs, y_enc, test_size=0.3, random_state=1)
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # X_over, y_over = oversample.fit_resample(X_train, y_train)

    (y_pred, X_proba), name = model_DT(X_train, X_test, y_train, y_test, feature_selected)
    print(feature_selected)
    print(X_test[0])
    print(y_test[0])
    acc = accuracy_score(y_test, y_pred)
    print(acc)

# perform the experiments for each drug.

def drug_data():
    file_name = './data/drug_consumption.data'
    print("Getting Data ...")
    data = get_data(file_name)
    #classes = [31, 28, 24, 20, 16, 29]
    classes = [31]
    for c in classes:
        convert_to_binary_class(data, c)

    X = data[:, 1:13]    # getting the features
    i = 1
    for c in classes:
        labels = data[:, c] #getting class label
        experiment(X, labels, i)
        i = i + 1

def main():
    drug_data()

if __name__ == "__main__":
    main()

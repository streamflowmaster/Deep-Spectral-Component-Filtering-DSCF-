from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def fetch_ml_model(model_name):
    if model_name == 'RandomForest':
        return RandomForestClassifier(random_state=0)
    elif model_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=0)
    elif model_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'GaussianNB':
        return GaussianNB()
    elif model_name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(100,50,30))
    elif model_name == 'SVM':
        return SVC(kernel='rbf',gamma='auto')
    else:
        raise NotImplementedError('Model name not found')
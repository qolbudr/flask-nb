from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

nb_model = GaussianNB()

def calculate_train_test_accuracies(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accuracies = []
    test_accuracies = []

    kfold_json = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        nb_model.fit(X_train, y_train)
        y_train_pred = nb_model.predict(X_train)
        y_test_pred = nb_model.predict(X_test)

        with open(f"./assets/model/gaussian_k{k}.pkl", "wb") as f:
            pickle.dump(nb_model, f)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print detailed accuracy per fold
        print(f'K-Fold {k}, Fold {fold}:')
        print(f'  Train Accuracy: {train_accuracy:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.4f}\n')
        
        json = {
            "fold": fold,
            "k": k,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

        kfold_json.append(json)


    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)

    return avg_train_accuracy, avg_test_accuracy, kfold_json

# Function to plot comparison of average accuracies across different K-Folds
def plot_accuracy_comparison(X, y, k_values):
    avg_train_accuracies = []
    avg_test_accuracies = []
    kfold_array = []

    for k in k_values:
        print(f'\n===== Evaluating K={k} =====')
        avg_train_acc, avg_test_acc, kfold_json = calculate_train_test_accuracies(X, y, k)
        kfold_array.append(kfold_json)
        avg_train_accuracies.append(avg_train_acc)
        avg_test_accuracies.append(avg_test_acc)

        # Print average accuracies for each K-Fold
        print(f'\nK={k} Average Train Accuracy: {avg_train_acc:.4f}')
        print(f'K={k} Average Test Accuracy: {avg_test_acc:.4f}')

    # Determine the best K-Fold
    best_k = k_values[np.argmax(avg_test_accuracies)]
    best_test_accuracy = max(avg_test_accuracies)
    print(f'\nThe K-Fold with the highest test accuracy is: K={best_k} with Test Accuracy={best_test_accuracy:.4f}')

    return avg_train_accuracies, avg_test_accuracies, best_k, kfold_array, best_test_accuracy
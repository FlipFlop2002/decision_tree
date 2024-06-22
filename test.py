import numpy as np
from my_tree_classes import *
from sklearn.model_selection import train_test_split
from utils import encode_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

file = 'tic-tac-toe.data'
X_encoded, y_encoded = encode_data(file)

# podział danych na zbiór treningowy i testowy
X_train, X_rest, y_train, y_rest = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=20, shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=20, shuffle=True)

best_depth = 1
prev_val_acc = 0
for i in range(1,10):
    DEPTH = i
    tree = DecisionTree(max_depth=DEPTH)
    tree.fit(X_train, y_train)

    y_pred_val = tree.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    print(f'depth: {i} -- val accuracy: {val_acc}')

    if val_acc > prev_val_acc:
        best_depth = i

    prev_val_acc = val_acc

# testowanie dla najleoszej głębokości
print(f'Best depth: {best_depth}')
tree = DecisionTree(max_depth=best_depth)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)


# obliczenie dokładności
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

plot_conf_matrix = True
if plot_conf_matrix:
    # macierz pomyłek
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    plt.title('Macierz Pomyłek')
    plt.colorbar()

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="black")

    plt.xlabel('przewidywana etykieta')
    plt.ylabel('rzeczywista etykieta')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.savefig(f'conf_matrix_depth_{DEPTH}_20.pdf')
    plt.show()
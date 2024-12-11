import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os

# Função de ativação
def sign(u):
    return 1 if u >= 0 else -1

# Normalização dos dados
def normalize(X):
    return 2 * (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0)) - 1

dim = 50
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

pessoas = list(sorted(os.listdir("./RecFac/")))
C = len(pessoas)

pasta_raiz = "./RecFac"

X = []
Y = []

for i, pessoa in enumerate(pessoas):
    
    for imagem in os.listdir(f"{pasta_raiz}/{pessoa}/"):

        img_original = cv2.imread(f"{pasta_raiz}/{pessoa}/{imagem}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img_original, (dim, dim))

        X.append(img.flatten())

        y = -np.ones(C)
        y[i] = 1

        Y.append(y)

X = np.array(X)
Y = np.array(Y)

X = normalize(X)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)

# Função MLP
def mlp(X, Y, hidden_layer = 10, max_epochs = 100, lr = 0.1):

    N, p = X.shape
    _, C = Y.shape

    W1 = np.random.random(size = (p, hidden_layer)) - 0.5
    b1 = np.zeros((1, hidden_layer))
    
    W2 = np.random.random(size = (hidden_layer, C)) - 0.5
    b2 = np.zeros((1, C))
    
    errors = []
    
    for epoch in range(max_epochs):
        
        total_error = 0
        
        for t in range(N):

            z1 = np.dot(X[t:t+1], W1) + b1
            a1 = tanh(z1)
            
            z2 = np.dot(a1, W2) + b2
            y_pred = tanh(z2)
            
            error = Y[t] - y_pred
            total_error += np.mean(np.abs(error))
            
            delta2 = error * tanh_derivative(z2)
            delta1 = np.dot(delta2, W2.T) * tanh_derivative(z1)
            
            W2 += lr * np.dot(a1.T, delta2)
            b2 += lr * delta2
            
            W1 += lr * np.dot(X[t:t+1].T, delta1)
            b1 += lr * delta1
            
        errors.append(total_error)
            
    return W1, b1, W2, b2, errors

def inference(X, W1, b1, W2, b2):

    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    
    z2 = np.dot(a1, W2) + b2
    y_pred = tanh(z2)

    return y_pred

# Função Perceptron
def perceptron(X, Y, max_epochs, lr):

    N, p = X.shape
    _, C = Y.shape

    X = np.hstack((-np.ones((N, 1)), X))
    W = np.zeros((C, p + 1))

    curve = np.zeros(max_epochs)

    for epoca in range(max_epochs):

        error = 0.0

        for t in range(len(X)):
            
            x_t = X[t]
            d_t = Y[t]

            u_t = W @ x_t
            y_t = np.array([sign(x) for x in u_t])
            
            e_t = d_t - y_t

            error += np.mean(np.abs(e_t))

            for j in range(C):
                W[j] += (lr * e_t[j] * x_t) / 2

        curve[epoca] = error

    return W, curve

def adaline(X, Y, max_epochs, lr):
    
    N, p = X.shape
    _, C = Y.shape

    X = np.hstack((-np.ones((N, 1)), X))
    W = np.zeros((C, p + 1))

    curve = np.zeros(max_epochs)

    for epoca in range(max_epochs):

        error = 0.0
        for t in range(len(X)):
            
            x_t = X[t]
            d_t = Y[t]

            u_t = W @ x_t
            y_t = u_t
            
            e_t = d_t - y_t

            error += np.mean(np.abs(e_t))

            for j in range(C):
                W[j] += lr * e_t[j] * x_t

        curve[epoca] = error

    return W, curve

# Validação Monte Carlo
index = np.arange(len(X))
train_size = int(0.8 * len(index))

# Listas para armazenar os resultados
results = { "Perceptron": [], "Adaline": [], "MLP": [] }
# results = { "Perceptron": [], "Adaline": [] }
# results = { "MLP": [] }

def output_perceptron(W, x):
    b = W[:, -1]
    W = W[:, : -1].copy()
    u_t = (W @ x) + b
    idx = np.argmax(u_t)
    out = -np.ones(W.shape[0])
    out[idx] *= -1.0
    return out

def calc_metrics(Y, y):

    n_samples, p = Y.shape
    cm = np.zeros((p, p))
    acc = 0

    for i in range(n_samples):
        
        idx_truth = np.argmax(Y[i])
        idx_predicted = np.argmax(y[i])

        if idx_truth == idx_predicted:
            acc += 1

        cm[idx_truth][idx_predicted] += 1

    acc = acc / n_samples

    return acc, cm

# Comparação de Modelos
for method, model in zip(["Perceptron", "Adaline", "MLP"], [perceptron, adaline, mlp]):
    
    for _ in range(50):
        
        np.random.shuffle(index)

        train_idx = index[:train_size]
        test_idx = index[train_size:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        if method == "MLP":
            
            W1, b1, W2, b2, errors = model(X_train, Y_train, hidden_layer = 20, max_epochs = 1000, lr = 0.001)
            
            Y_pred = np.array([inference(x, W1, b1, W2, b2) for x in X_test])

        elif method == "Adaline":

            W, errors = model(X_train, Y_train, max_epochs = 100, lr = 0.001)
            Y_pred = np.array([output_perceptron(W, x) for x in X_test])

        else:

            W, errors = model(X_train, Y_train, max_epochs = 100, lr = 0.01)
            Y_pred = np.array([output_perceptron(W, x) for x in X_test])

        accuracy, cm = calc_metrics(Y_test, Y_pred)
        results[method].append((accuracy, cm, errors))

# Exibe as estatísticas para cada modelo
for method, metrics in results.items():

    accuracies = [m[0] for m in metrics]
    confusions = [m[1] for m in metrics]
    curves = [m[2] for m in metrics]

    print(f"\n{method} Results:")

    if len(accuracies) == 0:
        print("Sem dados.")
        continue

    idx_max = np.argmax(accuracies)
    idx_min = np.argmin(accuracies)

    print(confusions[idx_max])

    plt.subplot(1, 2, 1)
    sns.heatmap(
        confusions[idx_max],
        annot=True, cmap="Greens", square=True,
        xticklabels=["1", "-1"], yticklabels=["1", "-1"], cbar=False
    )
    plt.title(f"Matriz de Confusão {method} - Maior Acurácia")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        confusions[idx_min],
        annot=True, cmap="Reds", square=True,
        xticklabels=["1", "-1"], yticklabels=["1", "-1"], cbar=False
    )
    plt.title(f"Matriz de Confusão {method} - Menor Acurácia")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    
    plt.figure(figsize=(8, 5))
    plt.plot(curves[idx_max], label="Maior Acurácia", color="Green")
    plt.plot(curves[idx_min], label="Menor Acurácia", color="Red")
    plt.title(f"Curva de Aprendizado - {method}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Total")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Média Acurácias: {np.mean(accuracies):.4f}")
    print(f"Desvio Padrão Acurácias: {np.std(accuracies):.4f}")
    print(f"Maior Valor Acurácias: {np.max(accuracies):.4f}")
    print(f"Menor Valor Acurácias: {np.min(accuracies):.4f}")

    print()
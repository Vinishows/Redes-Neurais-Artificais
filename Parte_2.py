import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Função de ativação
def sign(u):
    return 1 if u >= 0 else -1

# Normalização dos dados
def normalize(X):
    return 2 * (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0)) - 1

#Carregar dados

#Dimensões da imagem. Você deve explorar esse tamanho de acordo com o solicitado no pdf.
dim = 30 #50 signica que a imagem terá 50 x 50 pixels. ?No trabalho é solicitado para que se investigue dimensões diferentes:
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

pessoas = list(sorted(os.listdir("./RecFac/")))

pasta_raiz = "./RecFac"

C = 20 # Esse é o total de classes 

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

# Função MLP
def mlp(X, Y, hidden_layer_sizes, max_epochs, lr):
    
    # np.random.seed(42)

    N, p = X.shape
    X = np.concatenate((-np.ones((N, 1)), X), axis=1)  # Adiciona bias

    # Inicializa pesos
    w_hidden = np.random.uniform(-0.5, 0.5, (p + 1, hidden_layer_sizes[0]))
    w_output = np.random.uniform(-0.5, 0.5, (hidden_layer_sizes[0] + 1, 1))
    errors = []

    for epoch in range(max_epochs):
        total_error = 0
        for t in range(N):
            x_t = X[t].reshape(1, -1)
            target = Y[t]

            # Forward pass
            hidden_input = x_t @ w_hidden
            hidden_output = np.tanh(hidden_input)
            hidden_output = np.concatenate(([-1], hidden_output.flatten()))  # Bias na camada oculta

            final_input = hidden_output @ w_output
            final_output = np.tanh(final_input)

            # Cálculo do erro
            error = target - final_output
            total_error += error ** 2

            # Backpropagation
            delta_output = error * (1 - final_output ** 2)
            delta_hidden = (1 - hidden_output[1:] ** 2) * (w_output[1:, 0] * delta_output)

            w_output += lr * delta_output * hidden_output.reshape(-1, 1)
            w_hidden += lr * np.outer(x_t.flatten(), delta_hidden)

        errors.append(total_error)

    return (w_hidden, w_output), errors

# Função Adaline
def adaline(X, Y, max_epochs, lr):
    
    N, p = X.shape
    _, C = Y.shape

    X = np.hstack((-np.ones((N, 1)), X))
    W = np.random.random_sample((C, p + 1)) - 0.5

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

# Função Perceptron
def perceptron(X, Y, max_epochs, lr):
    
    N, p = X.shape
    _, C = Y.shape

    X = np.hstack((-np.ones((N, 1)), X))
    W = np.random.random_sample((C, p + 1)) - 0.5

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
                W[j] += lr * ((e_t[j] * x_t) / 2)

        curve[epoca] = error

    return W, curve

# Validação Monte Carlo
index = np.arange(len(X))
train_size = int(0.8 * len(index))

# Listas para armazenar os resultados
# results = { "Perceptron": [], "Adaline": [], "MLP": [] }
results = { "Perceptron": [], "Adaline": [] }

def output_perceptron(W, x):
    b = W[:, -1]
    W = W[:, : -1].copy()
    u_t = (W @ x) + b
    idx = np.argmax(u_t)
    out = -np.ones(W.shape[0])
    out[idx] *= -1.0
    return out

def calc_metrics(Y, y):

    N, p = Y.shape
    n_samples = Y.shape[0]
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
# for method, model in zip(["Perceptron", "Adaline", "MLP"], [perceptron, adaline, mlp]):
for method, model in zip(["Perceptron", "Adaline"], [perceptron, adaline]):
    
    for _ in range(5):
        
        np.random.shuffle(index)

        train_idx = index[:train_size]
        test_idx = index[train_size:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        if method == "MLP":
            
            (w_hidden, w_output), errors = model(X_train, Y_train, hidden_layer_sizes = [10], max_epochs=100, lr=0.01)

            X_test_bias = np.concatenate((-np.ones((X_test.shape[0], 1)), X_test), axis=1)
            hidden_outputs = np.tanh(X_test_bias @ w_hidden)
            hidden_outputs = np.concatenate((-np.ones((hidden_outputs.shape[0], 1)), hidden_outputs), axis=1)  # Bias
            
            Y_pred = np.array([sign(o) for o in (hidden_outputs @ w_output)])

        else:

            W, errors = model(X_train, Y_train, max_epochs = 10, lr = 1.0)
            Y_pred = np.array([output_perceptron(W, x) for x in X_test])

        accuracy, cm = calc_metrics(Y_test, Y_pred)
        results[method].append((accuracy, cm))

        # Calcula métricas
        # cm, accuracy, sensitivity, specificity = calc_metrics(Y_test.flatten(), Y_pred)
        # results[method].append((accuracy, sensitivity, specificity))

# Exibe as estatísticas para cada modelo
for method, metrics in results.items():

    accuracies = [m[0] for m in metrics]

    print(f"\n{method} Results:")
    print(f"Média Acurácias: {np.mean(accuracies):.4f}")
    print(f"Desvio Padrão Acurácias: {np.std(accuracies):.4f}")
    print(f"Maior Valor Acurácias: {np.max(accuracies):.4f}")
    print(f"Menor Valor Acurácias: {np.min(accuracies):.4f}")
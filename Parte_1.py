import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Função de ativação
def sign(u):
    return 1 if u >= 0 else -1

# Normalização dos dados
def normalize(X):
    return 2 * (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0)) - 1

# Carregar dados
data = np.loadtxt('spiral.csv', delimiter=",")
X = data[:, :2]
Y = data[:, 2:]

# Normalização
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
    
    X = X.T
    Y = Y.T
    p, N = X.shape

    X = np.concatenate((-np.ones((1, N)), X))
    errors = np.zeros((max_epochs,))
    w = np.random.random_sample((p + 1, 1)) - 0.5

    for epoca in range(max_epochs):
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = u_t
            d_t = Y[0, t]
            e_t = d_t - y_t
            errors[epoca] += np.abs(e_t)
            w = w + lr * e_t * x_t

    return w, errors

# Função Perceptron
def perceptron(X, Y_test, max_epochs, lr):
    
    X = X.T
    Y_ = Y_test.T
    p, N = X.shape
    
    X = np.concatenate((-np.ones((1, N)), X))
    errors = np.zeros((max_epochs,))
    w = np.random.random_sample((p + 1, 1)) - 0.5

    for epoca in range(max_epochs):

        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = float(Y_[0, t])
            e_t = d_t - y_t
            w = w + (lr * e_t * x_t) / 2

            errors[epoca] += np.abs(e_t)

    return w, errors

# Validação Monte Carlo
index = np.arange(len(data))
train_size = int(0.8 * len(index))

# Listas para armazenar os resultados
results = { "Perceptron": [], "Adaline": [], "MLP": [] }
# results = { "Perceptron": [], "Adaline": [] }

def inference_perceptron(w, x):
    b = w[-1]
    w = w[: -1].copy()
    w = w.reshape((w.shape[0], 1))
    return sign((w.T @ x) + b)

# Funções auxiliares para cálculo de sensibilidade e especificidade
def calc_metrics(Y_true, Y_pred):
    TP = np.sum((Y_true == 1) & (Y_pred == 1))
    TN = np.sum((Y_true == -1) & (Y_pred == -1))
    FP = np.sum((Y_true == -1) & (Y_pred == 1))
    FN = np.sum((Y_true == 1) & (Y_pred == -1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return np.array([[TP, FP], [FN, TN]]), accuracy, sensitivity, specificity

# Comparação de Modelos
for method, model in zip(["Perceptron", "Adaline", "MLP"], [perceptron, adaline, mlp]):
# for method, model in zip(["Perceptron", "Adaline"], [perceptron, adaline]):
    for _ in range(5):  # Monte Carlo com 10 execuções
        np.random.shuffle(index)
        train_idx = index[:train_size]
        test_idx = index[train_size:]

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        if method == "MLP":
            (w_hidden, w_output), curve = model(X_train, Y_train, hidden_layer_sizes=[4], max_epochs=100, lr=0.01)

            # Adiciona bias e realiza as predições
            X_test_bias = np.concatenate((-np.ones((X_test.shape[0], 1)), X_test), axis=1)
            hidden_outputs = np.tanh(X_test_bias @ w_hidden)
            hidden_outputs = np.concatenate((-np.ones((hidden_outputs.shape[0], 1)), hidden_outputs), axis=1)  # Bias
            Y_pred = np.array([sign(o) for o in (hidden_outputs @ w_output)])
        else:
            w, curve = model(X_train, Y_train, max_epochs = 10, lr = 0.001)
            Y_pred = np.array([sign((w[:-1].T @ x + w[-1]).item()) for x in X_test])

        # Calcula métricas
        cm, accuracy, sensitivity, specificity = calc_metrics(Y_test.flatten(), Y_pred)
        results[method].append((accuracy, sensitivity, specificity, cm, curve))

# Exibe as estatísticas para cada modelo
for method, metrics in results.items():

    accuracies = [m[0] for m in metrics]
    sensitivities = [m[1] for m in metrics]
    specificities = [m[2] for m in metrics]
    confusions = [m[3] for m in metrics]
    curves = [m[4] for m in metrics]

    idx_max = np.argmax(accuracies)
    idx_min = np.argmin(accuracies)

    sns.heatmap(
        confusions[idx_max],
        annot = True, fmt = "d", cmap = "Blues", square = True,
        xticklabels = ["1", "-1"], yticklabels = ["1", "-1"], cbar = False
    )
    plt.title(f"Matriz de Confusão {method} - Maior Acurácia")

    sns.heatmap(
        confusions[idx_min],
        annot = True, fmt = "d", cmap = "Reds", square = True,
        xticklabels = ["1", "-1"], yticklabels = ["1", "-1"], cbar = False
    )
    plt.title(f"Matriz de Confusão {method} - Menor Acurácia")

    #Grafico da Curva de Aprendizado
    # plt.plot(curves[idx_max])
    # plt.plot(curves[idx_min])

    print(f"\n{method} Results:")
    print(f"Média Acurácias: {np.mean(accuracies):.4f}")
    print(f"Desvio Padrão Acurácias: {np.std(accuracies):.4f}")
    print(f"Maior Valor Acurácias: {np.max(accuracies):.4f}")
    print(f"Menor Valor Acurácias: {np.min(accuracies):.4f}")

    print(f"Média Sensibilidade: {np.mean(sensitivities):.4f}")
    print(f"Desvio Padrão Sensibilidade: {np.std(sensitivities):.4f}")
    print(f"Maior Valor Sensibilidade: {np.max(sensitivities):.4f}")
    print(f"Menor Valor Sensibilidade: {np.min(sensitivities):.4f}")

    print(f"Média Especficidade: {np.mean(specificities):.4f}")
    print(f"Desvio Padrão Especficidade: {np.std(specificities):.4f}")
    print(f"Maior Valor Especficidade: {np.max(specificities):.4f}")
    print(f"Menor Valor Especficidade: {np.min(specificities):.4f}")
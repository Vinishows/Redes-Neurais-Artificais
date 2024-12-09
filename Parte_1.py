import numpy as np
import matplotlib.pyplot as plt

# Função de ativação
def sign(u):
    return 1 if u >= 0 else -1

# Normalização dos dados
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Carregar dados
data = np.loadtxt('spiral.csv', delimiter=",")
X = data[:, :2]
Y = data[:, 2:]

# Normalização
X = normalize(X)

# Função MLP
def mlp(X, Y, hidden_layer_sizes, max_epochs, lr):
    np.random.seed(42)
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
def adaline(X, Y, max_epoca, lr):
    X = X.T
    Y = Y.T
    p, N = X.shape
    X = np.concatenate((-np.ones((1, N)), X))
    errors = np.zeros((max_epoca,))
    w = np.random.random_sample((p + 1, 1)) - 0.5

    for epoca in range(max_epoca):
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
def perceptron(X, Y, max_epoca, lr):
    X = X.T
    Y = Y.T
    p, N = X.shape
    X = np.concatenate((-np.ones((1, N)), X))
    errors = np.zeros((max_epoca,))
    w = np.random.random_sample((p + 1, 1)) - 0.5

    for epoca in range(max_epoca):
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = float(Y[0, t])
            e_t = d_t - y_t
            errors[epoca] += np.abs(e_t)
            w = w + (lr * e_t * x_t) / 2

    return w, errors

# Validação Monte Carlo
index = np.arange(len(data))
train_size = int(0.8 * len(index))

# Listas para armazenar as acurácias
accuracy_results = {"Perceptron": [], "Adaline": [], "MLP": []}

# Comparação de Modelos
for method, model in zip(["Perceptron", "Adaline", "MLP"], [perceptron, adaline, mlp]):
    for _ in range(500):  # Monte Carlo com 500 execuções
        np.random.shuffle(index)
        train_idx = index[:train_size]
        test_idx = index[train_size:]

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        if method == "MLP":
            (w_hidden, w_output), errors = model(X_train, Y_train, hidden_layer_sizes=[4], max_epochs=10, lr=0.01)

            # Adiciona bias e realiza as predições
            X_test_bias = np.concatenate((-np.ones((X_test.shape[0], 1)), X_test), axis=1)
            hidden_outputs = np.tanh(X_test_bias @ w_hidden)
            hidden_outputs = np.concatenate((-np.ones((hidden_outputs.shape[0], 1)), hidden_outputs), axis=1)  # Bias
            Y_pred = np.array([sign(o) for o in (hidden_outputs @ w_output)])
        else:
            w, errors = model(X_train, Y_train, max_epoca=10, lr=0.01)
            Y_pred = np.array([sign((w[:-1].T @ x + w[-1]).item()) for x in X_test])

        # Calcula a acurácia e armazena
        accuracy = np.mean(Y_pred == Y_test.flatten())
        accuracy_results[method].append(accuracy)

# Funções auxiliares para cálculos manuais
def calc_mean(data):
    return sum(data) / len(data)

def calc_std(data, mean_value):
    return (sum((x - mean_value) ** 2 for x in data) / len(data)) ** 0.5

# Exibe as estatísticas para cada modelo
for method, accuracies in accuracy_results.items():
    mean_accuracy = calc_mean(accuracies)
    std_accuracy = calc_std(accuracies, mean_accuracy)
    print(f"\n{method} Results:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")
    print(f"Highest Accuracy: {max(accuracies):.4f}")
    print(f"Lowest Accuracy: {min(accuracies):.4f}")
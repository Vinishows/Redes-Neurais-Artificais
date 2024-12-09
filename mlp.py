import numpy as np

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

# Carregar dados e normalizar
data = np.loadtxt('spiral.csv', delimiter=",")
X = (data[:, :2] - np.mean(data[:, :2], axis=0)) / np.std(data[:, :2], axis=0)
Y = data[:, 2:]

# Configuração do MLP
accuracy_results = []
index = np.arange(len(data))
train_size = int(0.8 * len(index))

# Execução Monte Carlo
for _ in range(10):
    np.random.shuffle(index)
    train_idx = index[:train_size]
    test_idx = index[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    (w_hidden, w_output), errors = mlp(X_train, Y_train, hidden_layer_sizes=[4], max_epochs=100, lr=0.01)

    # Adiciona bias e realiza as predições
    X_test_bias = np.concatenate((-np.ones((X_test.shape[0], 1)), X_test), axis=1)
    hidden_outputs = np.tanh(X_test_bias @ w_hidden)
    hidden_outputs = np.concatenate((-np.ones((hidden_outputs.shape[0], 1)), hidden_outputs), axis=1)  # Bias
    Y_pred = np.array([1 if o >= 0 else -1 for o in (hidden_outputs @ w_output)])

    accuracy = np.mean(Y_pred == Y_test.flatten())
    accuracy_results.append(accuracy)

# Resultados
print("\nMLP Results:")
print(f"Mean Accuracy: {np.mean(accuracy_results):.4f}")
print(f"Standard Deviation: {np.std(accuracy_results):.4f}")
print(f"Highest Accuracy: {max(accuracy_results):.4f}")
print(f"Lowest Accuracy: {min(accuracy_results):.4f}")
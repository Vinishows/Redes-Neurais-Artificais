import numpy as np
import matplotlib.pyplot as plt

def sign(u):
    if u >= 0:
        return 1
    return -1

data = np.loadtxt('spiral.csv', delimiter = ",")

X = data[:, : 2]
Y = data[:, 2 :]

#Visualização dos dados:
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],s=90,marker='*',color='blue',label='Classe +1')
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],s=90,marker='s',color='red',label='Classe -1')
plt.legend()
#plt.show()

def mlp(X, Y, hidden_layer_sizes, max_epochs, lr):
    np.random.seed(42)
    N, p = X.shape
    X = np.concatenate((-np.ones((N, 1)), X), axis=1)  # Adiciona o bias como uma entrada adicional

    # Inicializa pesos para a camada oculta e de saída
    w_hidden = np.random.uniform(-0.5, 0.5, (p + 1, hidden_layer_sizes[0]))
    w_output = np.random.uniform(-0.5, 0.5, (hidden_layer_sizes[0] + 1, 1))

    errors = []

    for epoch in range(max_epochs):
        total_error = 0
        for t in range(N):
            # Forward pass
            x_t = X[t].reshape(1, -1)
            target = Y[t]

            # Ativação da camada oculta
            hidden_input = x_t @ w_hidden
            hidden_output = np.tanh(hidden_input)  # Função de ativação tangente hiperbólica
            hidden_output = np.concatenate(([-1], hidden_output.flatten()))  # Adiciona bias na saída oculta

            # Ativação da camada de saída
            final_input = hidden_output @ w_output
            final_output = np.tanh(final_input)

            # Cálculo do erro
            error = target - final_output
            total_error += error**2

            # Backpropagation
            delta_output = error * (1 - final_output**2)  # Derivada de tanh
            delta_hidden = (1 - hidden_output[1:]**2) * (w_output[1:, 0] * delta_output)

            # Atualização dos pesos
            w_output += lr * delta_output * hidden_output.reshape(-1, 1)
            w_hidden += lr * delta_hidden.reshape(1, -1).T @ x_t

        errors.append(total_error)

    return (w_hidden, w_output), errors


def adaline(X, Y, max_epoca, lr):
    
    #Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
    X = X.T
    Y = Y.T
    p, N = X.shape
    
    X = np.concatenate((
        -np.ones((1, N)),
        X)
    )
    
    errors = np.zeros((max_epoca, ))
    w = np.zeros((3, 1))
    w = np.random.random_sample((3, 1)) - .5

    for epoca in range(max_epoca):
        for t in range(N):
            x_t = X[:,t].reshape(p+1,1)
            u_t = (w.T @ x_t)[0,0]
            y_t = u_t
            d_t = Y[0, t]
            e_t = d_t - y_t
            errors[epoca] += np.abs(e_t)
            w = w + lr*e_t*x_t
    
    return w, errors


#Modelo do Perceptron Simples:
def perceptron(X, Y, max_epoca, lr):
    
    #Inicialização dos parâmetros (pesos sinápticos e limiar de ativação):
    X = X.T
    Y = Y.T
    p, N = X.shape
    
    X = np.concatenate((
        -np.ones((1, N)),
        X)
    )
    
    errors = np.zeros((max_epoca, ))
    w = np.zeros((3, 1))
    w = np.random.random_sample((3, 1)) - .5

    for epoca in range(max_epoca):
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T@x_t)[0,0]
            y_t = sign(u_t)
            d_t = float(Y[0, t])
            e_t = d_t - y_t
            errors[epoca] += np.abs(e_t)
            w = w + (lr*e_t*x_t)/2
    
    return w, errors

def inference_perceptron(w, x):
    b = w[-1]
    w = w[: -1].copy()
    w = w.reshape((w.shape[0], 1))
    return sign((w.T @ x) + b)

# print(inference_perceptron(w, np.array([30, 30])))

def confusion_matrix(Y, y):
    tp = np.sum((Y.flatten() == 1) & (y.flatten() == 1))
    fp = np.sum((Y.flatten() == 1) & (y.flatten() == -1))
    tn = np.sum((Y.flatten() == -1) & (y.flatten() == -1))
    fn = np.sum((Y.flatten() == -1) & (y.flatten() == 1))

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensibility = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return np.array([[tp, fp], [fn, tn]]), accuracy, sensibility, specificity

index = np.arange(len(data))

treno = int(0.8 * len(index))

#
#  True Positive    False Positive
#  False Negative   True Negative
#

matrizes = []
acuracias = []
sensibilidades = []
especificidades = []

for _ in range(50):
    
    np.random.shuffle(index)

    treno_idx = index[: treno]
    tete_idx = index[treno :]

    X_treino, X_teste = X[treno_idx], X[tete_idx]
    y_treino, y_teste = Y[treno_idx], Y[tete_idx]

    w, errors = adaline(X_treino, y_treino, max_epoca = 10, lr = 0.001)

    y_pred = np.array([inference_perceptron(w, x) for x in X_teste])

    cm, acc, sens, spec = confusion_matrix(y_teste, y_pred)

    matrizes.append(cm)
    acuracias.append(acc)
    sensibilidades.append(sens)
    especificidades.append(spec)

matrizes = np.array(matrizes)
acuracias = np.array(acuracias)
sensibilidades = np.array(sensibilidades)
especificidades = np.array(especificidades)

# TP = np.sum((Y.flatten() == 1) & (y.flatten() == 1))
# FP = np.sum((Y.flatten() == 1) & (y.flatten() == -1))
# TN = np.sum((Y.flatten() == -1) & (y.flatten() == -1))
# FN = np.sum((Y.flatten() == -1) & (y.flatten() == 1))

print("Média das Acurácias:", acuracias.mean())
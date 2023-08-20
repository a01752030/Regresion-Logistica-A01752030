import numpy as np
from Data import GetData

#Agarra datos
xy=GetData()

X=xy[0]
y=xy[1]


#Funcion sigmoidal
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Escalamiento
def feature_scaling(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

#Funcion de costo
def cost(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + reg_term

#Gradiente descendiente
def gradient_descent(X, y, theta, alpha, num_iterations, lambda_):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        gradient = (1/m) * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
        gradient[1:] += (lambda_ / m) * theta[1:]  # Regularization for theta[1:]
        theta -= alpha * gradient
        cost_history.append(cost(X, y, theta, lambda_))

    return theta, cost_history

#Predicción
def predict(X, theta):
    h = sigmoid(np.dot(X, theta))
    return [1 if i >= 0.5 else 0 for i in h]

#Efectividad de prediccion
def acc(predictions, actual):
    correct = sum(p == a for p, a in zip(predictions, actual))
    return (correct / len(actual)) * 100

#Bias
X = np.hstack((np.ones((X.shape[0], 1)), X))

#Parametros
theta = np.zeros(X.shape[1])
alpha = 0.01
num_iterations = 1000
lambda_ = 0.1  # Regularization parameter

#Entrenamiento

theta, cost_history = gradient_descent(X, y, theta, alpha, num_iterations, lambda_)



if __name__=="__main__":
    predictions = predict(X, theta)
    print("Predicciones:", predictions)
    print("Valores reales:", y.tolist())
    acc = acc(predictions, y)
    print(f"Accuracy: {acc:.2f}%")
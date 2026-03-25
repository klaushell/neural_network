import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(a):
    return 1 - a**2


class XORNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, seed=42):
        np.random.seed(seed)

        # More stable initialization
        limit1 = np.sqrt(6 / (input_size + hidden_size))
        limit2 = np.sqrt(6 / (hidden_size + output_size))

        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = tanh(self.z1)  # often better than sigmoid in the hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # output between 0 and 1
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Sigmoid output for binary classification
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        delta1 = np.dot(delta2, self.W2.T) * tanh_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, num_epochs=20000, learning_rate=0.1):
        for epoch in range(num_epochs):
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 2000 == 0:
                eps = 1e-9
                loss = -np.mean(
                    y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
                )
                print(f"Epoch {epoch}: loss = {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


def xor_dataset():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def run_numpy(X, y):
    nn = XORNeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=42)
    nn.train(X, y, num_epochs=20000, learning_rate=0.1)

    prediction = nn.predict(X)
    rounded = np.round(prediction, decimals=0)

    print("\nRaw predictions:")
    print(prediction)

    print("\nRounded predictions:")
    print(rounded)


def run_keras(X, y):
    try:
        import keras
        from keras.layers import Dense
        from keras.models import Sequential
    except ImportError as e:
        raise ImportError(
            "Keras path requires the 'keras' and 'tensorflow' packages. "
            "Install with: pip install -r requirements.txt"
        ) from e

    keras.utils.set_random_seed(42)
    np.random.seed(42)

    model = Sequential()
    model.add(Dense(8, input_dim=2, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

    prediction = model.predict(X, verbose=0)
    rounded = np.round(prediction, decimals=0)

    print("\nRaw predictions:")
    print(prediction)

    print("\nRounded predictions:")
    print(rounded)

    loss, accuracy = model.evaluate(X, y, verbose=0)
    print("\nFinal loss:", loss)
    print("Final accuracy:", accuracy)


def main():
    X, y = xor_dataset()

    answer = input("Use Keras instead of NumPy? [y/N]: ").strip().lower()
    use_keras = answer in ("y", "yes")

    if use_keras:
        run_keras(X, y)
    else:
        run_numpy(X, y)


if __name__ == "__main__":
    main()

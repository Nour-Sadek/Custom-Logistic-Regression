import numpy as np


class CustomLogisticRegressionClass:

    """A custom Logistic Regression class that does logistic regression on a numpy 2d array of the shape
    n_observations x n_features. After training on a training set, it would be able to predict binary class affiliation.

    === Private Attributes ===
    _learning_rate: a hyperparameter that controls how much the model updates the weights during each trng cycle
    _iterations: the number of times the algorithm goes through the training data set
    _weights: set of coefficients assigned to each feature that quantify their effect in class membership
    _bias: intercept term

    """

    _learning_rate: float
    _iterations: int
    _threshold: float
    _weights: np.ndarray
    _bias: float
    _history: dict

    def __init__(self, learning_rate: float, iterations: int, threshold: float = 0.5) -> None:
        """Initialize a new CustomLogisticRegressionClass with _learning_rate <learning_rate> and
        _iterations <iterations>, and a threshold <threshold> of determining class belonging where class = 1
        if prob > threshold anf class = 0 otherwise. The default for threshold is 0.5."""

        self._learning_rate = learning_rate
        self._iterations = iterations
        self._threshold = threshold
        self._weights = None
        self._bias = None
        self._history = {"Accuracy": [], "Precision": [], "Recall": [], "loss": []}

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        """Return a 1-D numpy array that transforms values in <X> using the transformation sigmoid function
        to values between 0 and 1."""
        transformed_values = 1 / (1 + np.exp(-X))
        return transformed_values

    def bce_loss(self, y: np.ndarray, y_predictions: np.ndarray) -> np.floating[any]:
        """Return the Binary Cross Entropy (BCE) loss between <y> and <y_predictions>"""
        bce_loss = - np.mean((y * np.log(y_predictions)) + ((1 - y) * np.log(1 - y_predictions)))
        return bce_loss

    def compute_metrics(self, y: np.ndarray, y_predictions: np.ndarray) -> tuple:
        """Return a 3-tuple of (accuracy, precision, recall) calculated based on the given and predicted classifications
        where:
        - accuracy is the proportion of how many predictions are accurate
        - precision is the proportion of how many of the positive predictions are correct
        - rcall is the proportion of how many true positives were correctly predicted"""
        true_positives = np.sum((y == 1) & (y_predictions == 1))
        true_negatives = np.sum((y == 0) & (y_predictions == 0))
        false_positives = np.sum((y == 0) & (y_predictions == 1))
        false_negatives = np.sum((y == 1) & (y_predictions == 0))

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return float(accuracy), float(precision), float(recall)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Update the <self._weights> and <self.bias> terms trained on the training set <X_train> of shape
        n_observations x n_features and the target variables (either 0 or 1) for each observation <y_train> using
        Binary Cross Entropy as the cost function and Gradient Descent as the optimization algorithm."""

        n_observations, n_features = X_train.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

        for _ in range(self._iterations):
            y_predictions = self.predict_probabilities(X_train)

            # Calculate the derivative of the mse_loss in terms of the weights and bias and update the terms
            d_weights = (1 / n_observations) * (X_train.T @ (y_predictions - y_train))
            self._weights = self._weights - self._learning_rate * d_weights
            d_bias = (1 / n_observations) * np.sum(y_predictions - y_train)
            self._bias = self._bias - self._learning_rate * d_bias

            # Update the <self._history> term
            bce_loss = self.bce_loss(y_train, y_predictions)
            metrics = self.compute_metrics(y_train, np.where(y_predictions > self._threshold, 1, 0))
            self._history["Accuracy"].append(metrics[0])
            self._history["Precision"].append(metrics[1])
            self._history["Recall"].append(metrics[2])
            self._history["loss"].append(float(bce_loss))

    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1-D numpy array of probabilities after applying the linear model on <X_test> and transforming
        the values using the sigmoid function"""
        model = np.dot(X_test, self._weights) + self._bias
        y_predictions = self.sigmoid(model)
        return y_predictions

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1-D numpy array of values 0 or 1 that predicts class affiliation of each observation in <X_test>
        of shape n_observations x n_features based on the trained model that uses <self._weights> and <self._bias>
        determined after running the self.fit method and based on threshold <self._threshold>."""
        y_predictions = self.predict_probabilities(X_test)
        return np.where(y_predictions > self._threshold, 1, 0)

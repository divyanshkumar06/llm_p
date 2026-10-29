import numpy as np
import scipy.sparse as sp

class CustomLogisticRegression:
    """
    Custom Multi-class Logistic Regression Classifier (One-vs-Rest / Softmax)
    built strictly from scratch with NumPy without any sklearn predefined classes.
    
    Trained using vectorized Mini-batch Gradient Descent with L2 regularization.
    """
    def __init__(self, learning_rate=0.5, n_epochs=25, batch_size=512, l2_reg=1e-4, verbose=True):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.verbose = verbose
        
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.class_to_idx_ = {}

    def _sigmoid(self, z):
        # Clip z for numerical stability
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        Fits the logistic regression weights using vectorized matrix operations.
        X: CSR sparse matrix or dense ndarray of shape (n_samples, n_features)
        y: Target label vector of shape (n_samples,)
        """
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Convert y to One-Hot Binary Target Matrix (n_samples, n_classes)
        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, label in enumerate(y):
            Y_onehot[i, self.class_to_idx_[label]] = 1.0
            
        # Initialize Weights (n_features, n_classes) and Biases (n_classes,)
        self.weights_ = np.zeros((n_features, n_classes), dtype=np.float32)
        self.bias_ = np.zeros(n_classes, dtype=np.float32)
        
        indices = np.arange(n_samples)
        
        if self.verbose:
            print(f"Training Custom Logistic Regression ({n_samples} samples, {n_features} features, {n_classes} classes)...")
            
        for epoch in range(1, self.n_epochs + 1):
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                Y_batch = Y_onehot[batch_indices]
                b_size = end_idx - start_idx
                
                # Forward pass: Z = X_batch * W + b
                Z = X_batch.dot(self.weights_) + self.bias_
                P = self._sigmoid(Z)
                
                # Error gradient
                error = P - Y_batch  # (batch_size, n_classes)
                
                # Gradients: dW = (1/b_size) * X_batch^T * error + l2_reg * W
                dW = (X_batch.T.dot(error) / b_size) + self.l2_reg * self.weights_
                db = np.mean(error, axis=0)
                
                # Weight update
                self.weights_ -= self.learning_rate * dW
                self.bias_ -= self.learning_rate * db
                
            if self.verbose and (epoch % 5 == 0 or epoch == self.n_epochs):
                # Compute training loss
                Z_all = X.dot(self.weights_) + self.bias_
                P_all = self._sigmoid(Z_all)
                # Binary Cross-Entropy Loss
                eps = 1e-12
                loss = -np.mean(Y_onehot * np.log(P_all + eps) + (1 - Y_onehot) * np.log(1 - P_all + eps))
                print(f"  Epoch {epoch:2d}/{self.n_epochs} | Loss: {loss:.4f}")
                
        return self

    def predict_proba(self, X):
        """
        Computes probability predictions for each class.
        """
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet!")
        Z = X.dot(self.weights_) + self.bias_
        P = self._sigmoid(Z)
        # Normalize per row to get proper probability distribution
        row_sums = np.sum(P, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return P / row_sums

    def predict(self, X):
        """
        Predicts the class label with highest probability.
        """
        proba = self.predict_proba(X)
        max_indices = np.argmax(proba, axis=1)
        return self.classes_[max_indices]

if __name__ == "__main__":
    # Test Custom Logistic Regression with synthetic data
    X_test = sp.csr_matrix(np.random.randn(100, 50))
    y_test = np.random.choice(["hin_Deva", "tam_Taml", "ben_Beng"], size=100)
    clf = CustomLogisticRegression(n_epochs=10)
    clf.fit(X_test, y_test)
    preds = clf.predict(X_test)
    print("Predictions shape:", preds.shape)
    print("Sample predictions:", preds[:5])

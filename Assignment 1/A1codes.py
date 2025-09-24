import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import os

def minimizeL2(X, y):
    """
    Solves L2 linear regression using closed-form solution:
    w = (X^T X)^(-1) X^T y
    
    Parameters:
    X : numpy.ndarray
        Input matrix of shape (n, d)
    y : numpy.ndarray
        Target vector of shape (n, 1)
    
    Returns:
    w : numpy.ndarray
        Weight vector of shape (d, 1)
    """
    # Compute (X^T X)
    XtX = X.T @ X
    
    # Compute (X^T y)
    Xty = X.T @ y
    
    # Solve XtX w = Xty for w
    w = np.linalg.solve(XtX, Xty)
    
    return w

def minimizeLinf(X, y):
    """
    Solves L∞ linear regression using linear programming

    Parameters:
    X : numpy.ndarray
        Input matrix of shape (n, d)
    y : numpy.ndarray
        Target vector of shape (n, 1)

    Returns:
    w : numpy.ndarray
        Weight vector of shape (d, 1)
    """

    n, d = X.shape

    # Decision variables: w (d x 1), delta (1 x 1)
    # u = [w; delta] ∈ R^{d+1}

    # Objective: min delta
    c = np.zeros((d + 1, 1))
    c[-1, 0] = 1  # Only delta is minimized

    # Constraints:
    # 1. delta >= 0
    G1 = np.zeros((1, d + 1))
    G1[0, -1] = -1
    h1 = np.zeros((1, 1))

    # 2. Xw - y <= delta * 1_n
    G2 = np.hstack([X, -np.ones((n, 1))])
    h2 = y

    # 3. y - Xw <= delta * 1_n
    G3 = np.hstack([-X, -np.ones((n, 1))])
    h3 = -y

    # Stack all constraints
    G = np.vstack([G1, G2, G3])
    h = np.vstack([h1, h2, h3])

    # Convert to cvxopt matrices
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    c_cvx = matrix(c)

    # Suppress solver output
    solvers.options['show_progress'] = False

    # Solve LP
    sol = solvers.lp(c_cvx, G_cvx, h_cvx)
    u_opt = np.array(sol['x']).reshape(-1, 1)
    w_opt = u_opt[:d]

    return w_opt

def synRegExperiments():
    """
    Synthetic regression experiments to compare L2 and L∞ regression.
    
    This function generates synthetic data and then applies both L2 and L∞
    regression methods to it. The losses for both training and testing
    datasets are computed and averaged over multiple runs.
    
    Returns:
    avg_train_loss : numpy.ndarray
        Average training loss for both models and metrics. Shape (2, 2).
    avg_test_loss : numpy.ndarray
        Average testing loss for both models and metrics. Shape (2, 2).
    """
    import numpy as np

    def genData(n_points, is_training=False):
        X = np.random.randn(n_points, d)  # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1)  # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise  # ground truth label
        if is_training:
            y[0] *= -0.1
        return X, y

    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2

    train_loss = np.zeros((n_runs, 2, 2))  # n_runs x n_models x n_metrics
    test_loss = np.zeros((n_runs, 2, 2))   # n_runs x n_models x n_metrics

    np.random.seed(42)  # Use a fixed seed for reproducibility

    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)

        # Train models
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Training losses
        train_pred_L2 = Xtrain @ w_L2
        train_pred_Linf = Xtrain @ w_Linf
        train_loss[r, 0, 0] = np.mean((train_pred_L2 - ytrain) ** 2)      # L2 model, L2 loss
        train_loss[r, 0, 1] = np.max(np.abs(train_pred_L2 - ytrain))      # L2 model, Linf loss
        train_loss[r, 1, 0] = np.mean((train_pred_Linf - ytrain) ** 2)    # Linf model, L2 loss
        train_loss[r, 1, 1] = np.max(np.abs(train_pred_Linf - ytrain))    # Linf model, Linf loss

        # Test losses
        test_pred_L2 = Xtest @ w_L2
        test_pred_Linf = Xtest @ w_Linf
        test_loss[r, 0, 0] = np.mean((test_pred_L2 - ytest) ** 2)         # L2 model, L2 loss
        test_loss[r, 0, 1] = np.max(np.abs(test_pred_L2 - ytest))         # L2 model, Linf loss
        test_loss[r, 1, 0] = np.mean((test_pred_Linf - ytest) ** 2)       # Linf model, L2 loss
        test_loss[r, 1, 1] = np.max(np.abs(test_pred_Linf - ytest))       # Linf model, Linf loss

    # Average over runs
    avg_train_loss = np.mean(train_loss, axis=0)  # shape (2, 2)
    avg_test_loss = np.mean(test_loss, axis=0)    # shape (2, 2)

    return avg_train_loss, avg_test_loss

def preprocessCCS(dataset_folder):
    """
    Loads and preprocesses the CCS dataset.

    Parameters:
    dataset_folder : str
        Absolute path to the folder containing Concrete_Data.xls

    Returns:
    X : numpy.ndarray
        Input matrix of shape (n, d)
    y : numpy.ndarray
        Target vector of shape (n, 1)
    """
    import pandas as pd

    # Load the dataset
    data_path = os.path.join(dataset_folder, "Concrete_Data.xls")
    df = pd.read_excel(data_path)

    # Last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    return X, y

def runCCS(dataset_folder):
    """
    Runs regression experiments on the CCS dataset.

    Parameters:
    dataset_folder : str
        Absolute path to the folder containing Concrete_Data.xls

    Returns:
    avg_train_loss : numpy.ndarray
        Average training loss for both models and metrics. Shape (2, 2).
    avg_test_loss : numpy.ndarray
        Average testing loss for both models and metrics. Shape (2, 2).
    """
    import numpy as np

    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)  # augment input

    n_runs = 100
    train_loss = np.zeros((n_runs, 2, 2))  # n_runs x n_models x n_metrics
    test_loss = np.zeros((n_runs, 2, 2))   # n_runs x n_models x n_metrics

    np.random.seed(101234289)

    for r in range(n_runs):
        # Randomly partition the dataset into 50% train, 50% test
        idx = np.random.permutation(n)
        n_train = int(0.5 * n)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        Xtrain, ytrain = X[train_idx], y[train_idx]
        Xtest, ytest = X[test_idx], y[test_idx]

        # Train models
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Training losses
        train_pred_L2 = Xtrain @ w_L2
        train_pred_Linf = Xtrain @ w_Linf
        train_loss[r, 0, 0] = np.mean((train_pred_L2 - ytrain) ** 2)      # L2 model, L2 loss
        train_loss[r, 0, 1] = np.max(np.abs(train_pred_L2 - ytrain))      # L2 model, Linf loss
        train_loss[r, 1, 0] = np.mean((train_pred_Linf - ytrain) ** 2)    # Linf model, L2 loss
        train_loss[r, 1, 1] = np.max(np.abs(train_pred_Linf - ytrain))    # Linf model, Linf loss

        # Test losses
        test_pred_L2 = Xtest @ w_L2
        test_pred_Linf = Xtest @ w_Linf
        test_loss[r, 0, 0] = np.mean((test_pred_L2 - ytest) ** 2)         # L2 model, L2 loss
        test_loss[r, 0, 1] = np.max(np.abs(test_pred_L2 - ytest))         # L2 model, Linf loss
        test_loss[r, 1, 0] = np.mean((test_pred_Linf - ytest) ** 2)       # Linf model, L2 loss
        test_loss[r, 1, 1] = np.max(np.abs(test_pred_Linf - ytest))       # Linf model, Linf loss

    avg_train_loss = np.mean(train_loss, axis=0)  # shape (2, 2)
    avg_test_loss = np.mean(test_loss, axis=0)    # shape (2, 2)

    return avg_train_loss, avg_test_loss

# quick test
def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[7], [8], [9]])
    
    w = minimizeL2(X, y)
    print("Computed weights w (L2):")
    print(w)

    w_inf = minimizeLinf(X, y)
    print("Computed weights w (Linf):")
    print(w_inf)

    # Run synthetic regression experiments
    avg_train_loss, avg_test_loss = synRegExperiments()
    print("Average training loss (L2, Linf):")
    print(avg_train_loss[0])
    print("Average testing loss (L2, Linf):")
    print(avg_test_loss[0])

if __name__ == "__main__":
    main()


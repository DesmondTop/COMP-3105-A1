import numpy as np
from cvxopt import matrix, solvers
import csv
import os
import xlrd

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

    np.random.seed(101234289)  # Use a fixed seed for reproducibility

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

def preprocessCCS(dataset_folder=None):
    """
    Loads and preprocesses the CCS dataset without pandas.

    Parameters:
    dataset_folder : str or None
        If None, uses the current directory. Otherwise, absolute path to folder containing Concrete_Data.xls.

    Returns:
    X : numpy.ndarray
        Input matrix of shape (n, d)
    y : numpy.ndarray
        Target vector of shape (n, 1)
    """

    # Use current directory if no folder is specified
    if dataset_folder is None:
        data_path = "Concrete_Data.xls"
    else:
        data_path = os.path.join(dataset_folder, "Concrete_Data.xls")

    # Open the workbook and select the first sheet
    wb = xlrd.open_workbook(data_path)
    sheet = wb.sheet_by_index(0)

    # Read all rows, skip header (row 0)
    data = []
    for row_idx in range(1, sheet.nrows):
        row = sheet.row_values(row_idx)
        data.append(row)

    data = np.array(data, dtype=float)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    return X, y

def runCCS(dataset_folder=None):
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


def test_toy_regression():
    """
    Loads toy regression data from toy_data folder, fits L2 and Linf models, and prints losses in table format.
    """
    # Use correct relative path to toy_data folder
    toy_folder = "../toy_data"
    train_path = os.path.join(toy_folder, "regression_train.csv")
    test_path = os.path.join(toy_folder, "regression_test.csv")

    # Load and augment training data
    with open(train_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        train_data = [row for row in reader]
    train_data = np.array(train_data, dtype=float)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

    # Load and augment test data
    with open(test_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        test_data = [row for row in reader]
    test_data = np.array(test_data, dtype=float)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

    # Fit models
    w_L2 = minimizeL2(X_train, y_train)
    w_Linf = minimizeLinf(X_train, y_train)

    # Training losses
    train_pred_L2 = X_train @ w_L2
    train_pred_Linf = X_train @ w_Linf
    train_L2_loss_L2 = np.mean((train_pred_L2 - y_train) ** 2)
    train_Linf_loss_L2 = np.max(np.abs(train_pred_L2 - y_train))
    train_L2_loss_Linf = np.mean((train_pred_Linf - y_train) ** 2)
    train_Linf_loss_Linf = np.max(np.abs(train_pred_Linf - y_train))

    # Test losses
    test_pred_L2 = X_test @ w_L2
    test_pred_Linf = X_test @ w_Linf
    test_L2_loss_L2 = np.mean((test_pred_L2 - y_test) ** 2)
    test_Linf_loss_L2 = np.max(np.abs(test_pred_L2 - y_test))
    test_L2_loss_Linf = np.mean((test_pred_Linf - y_test) ** 2)
    test_Linf_loss_Linf = np.max(np.abs(test_pred_Linf - y_test))

    # Print model weights
    print("\nModel weights:")
    print("w_L2 =\n", w_L2)
    print("w_Linf =\n", w_Linf)

    # Print results in table format
    print("\nTable 1: Training losses")
    print("Model      | L2 loss      | Linf loss")
    print(f"L2 model   | {train_L2_loss_L2:.8f} | {train_Linf_loss_L2:.8f}")
    print(f"Linf model | {train_L2_loss_Linf:.8f} | {train_Linf_loss_Linf:.8f}")

    print("\nTable 2: Test losses")
    print("Model      | L2 loss      | Linf loss")
    print(f"L2 model   | {test_L2_loss_L2:.8f} | {test_Linf_loss_L2:.8f}")
    print(f"Linf model | {test_L2_loss_Linf:.8f} | {test_Linf_loss_Linf:.8f}")



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

    # Run toy regression testing using files in toy_data
    print("\n--- Toy Regression File Testing ---")
    test_toy_regression()

    # Run CCS regression testing using Concrete_Data.xls
    print("\n--- CCS Regression File Testing ---")
    avg_train_loss, avg_test_loss = runCCS()  # No argument needed if file is in current directory

    print("\nTable 3: CCS Training losses (averaged over 100 runs)")
    print("Model      | L2 loss      | Linf loss")
    print(f"L2 model   | {avg_train_loss[0,0]:.8f} | {avg_train_loss[0,1]:.8f}")
    print(f"Linf model | {avg_train_loss[1,0]:.8f} | {avg_train_loss[1,1]:.8f}")

    print("\nTable 4: CCS Test losses (averaged over 100 runs)")
    print("Model      | L2 loss      | Linf loss")
    print(f"L2 model   | {avg_test_loss[0,0]:.8f} | {avg_test_loss[0,1]:.8f}")
    print(f"Linf model | {avg_test_loss[1,0]:.8f} | {avg_test_loss[1,1]:.8f}")

if __name__ == "__main__":
    main()
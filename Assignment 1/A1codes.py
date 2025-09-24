import numpy as np

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

# quick test
def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[7], [8], [9]])
    
    w = minimizeL2(X, y)
    print("Computed weights w:")
    print(w)

if __name__ == "__main__":
    main()


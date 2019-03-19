from sklearn.utils.extmath import randomized_svd
import numpy as np
import pandas as pd
import sys

import classes as c

def project_label(y_true, U):
    """
    Project training labels y_true into a subspace using a precomputed SVD U matrix.
    """
    return np.matmul(U.T, y_true)

def decode_pred(y_pred, U):
    """
    Reverse the PLST transform by computing round(U @ y_pred).
    """
    return np.rint(np.matmul(U, y_pred))

n_components = 75
def __plst_main():
    print("Loading metadata...")
    df = pd.read_csv(sys.argv[1], index_col='id', nrows=50000)
    
    filtered = df.filter(items=c.all_tags)
    X = filtered.values.T
    
    print(filtered.head())
    print("Performing SVD...")
    U, Sigma, VT = randomized_svd(X, 
                                  n_components=n_components,
                                  n_iter=5,
                                  random_state=None)
    
    print("Component explained variances:")
    sum_sigma = np.sum(np.power(Sigma, 2))
    explained_variance = np.zeros(n_components)
    cur_total = 0
    for i in range(n_components):
        explained_variance[i] = (Sigma[i]**2) / sum_sigma
        cur_total += explained_variance[i]
        print("    {}: {:.3f} / {:.3f}".format(i, explained_variance[i], cur_total))
        
    print("Total explained variance: {}".format(np.sum(explained_variance))) 
    np.save('./plst_u.npy', U)

if __name__ == '__main__':
    __plst_main()

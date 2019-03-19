import numpy as np
import pandas as pd
import sys

import classes as c

N = 10000
def main():
    print("Loading metadata...")
    df = pd.read_csv(sys.argv[1], index_col='id', nrows=50000)
    
    filtered = df.filter(items=c.all_tags)
    U = np.load('./plst_u.npy')
    
    print("Projecting labels...")
    projected = filtered.aggregate(lambda row: np.dot(U.T, row.values), axis=1, result_type='expand')
    comp_min = np.amin(projected.values, axis=0)
    comp_max = np.amax(projected.values, axis=0)
    
    print("Resampling.")
    s = np.random.uniform(comp_min, comp_max, size=[N, U.shape[-1]])
    nearest_neighbors = []
    for i in range(N):
        dist = np.sqrt(np.sum((s[i] - projected.values)**2, axis=-1))
        nn = np.argmin(dist)
        nearest_neighbors.append(nn)
        
        if (i+1) % 50 == 0:
            print("    {} of {}...".format(i+1, N))
        
    resampled = df.iloc[nearest_neighbors]
    print(resampled.head())
    
    resampled.to_csv(sys.argv[2])

if __name__ == '__main__':
    main()

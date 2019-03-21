import pandas as pd
import numpy as np

import sys
import classes

def calculate_ir(filtered_df, cloned_samples):
    df_counts = np.sum(filtered_df.values, axis=0)
    
    if cloned_samples is not None:
        cloned_df = filtered_df.loc[cloned_samples]
        cloned_counts = np.sum(cloned_df.values, axis=0)
        total_counts = df_counts + cloned_counts
    else:
        total_counts = df_counts
    
    max = np.amax(total_counts)
    return max / total_counts

def ml_random_oversampling(df, all_labels, oversample_pct):
    cloned_samples = []
    samples_to_clone = int(len(df) * oversample_pct)
    
    irbl = calculate_ir(df, None)
    mean_ir = np.mean(irbl)
    
    oversample_labels = []
    for i, label in enumerate(all_labels):
        if irbl[i] > mean_ir:
            oversample_labels.append(label)
    
    print("MeanIR = {:.3f}".format(mean_ir))
    print("Cloning samples with labels: {}".format(', '.join(map(lambda l: "'"+l+"'", oversample_labels))))
    iter = 0
    while len(cloned_samples) < samples_to_clone and len(oversample_labels) > 0:
        for label in oversample_labels:
            bag = df[df[label] == 1]
            cloned = bag.sample(1)
            cloned_samples.append(cloned.index[0])
    
        if iter % 10 == 0:
            print("Iteration {}: cloned {} samples ({} / {})".format(iter, len(oversample_labels), len(cloned_samples), samples_to_clone))

        irbl = calculate_ir(df, cloned_samples)
        for i, label in enumerate(all_labels):
            if label not in oversample_labels:
                continue
            
            if irbl[i] <= mean_ir:
                oversample_labels.remove(label)
                print("Iteration {}: excluding label '{}' from cloning".format(iter, label))

        if iter % 10 == 0:
            print("Iteration {}: new MeanIR = {:.3f} (threshold={:.3f})".format(iter, np.mean(irbl), mean_ir))
        
        iter += 1
        
    return df.loc[cloned_samples]

def main():
    df = pd.read_csv(sys.argv[1], index_col='id')
    filtered = df.filter(items=classes.all_tags)
    
    labels = []
    sample_counts = np.sum(filtered.values, axis=0)
    for idx, label in enumerate(classes.all_tags):
        if sample_counts[idx] < 750:
            print("Excluding label {} (has only {} samples)".format(label, sample_counts[idx]))
        else:
            labels.append(label)
            
    filtered = df.filter(items=labels)
    oversample_pct = float(sys.argv[2])
    cloned_df = ml_random_oversampling(filtered, labels, oversample_pct)
    
    cloned_df.to_csv(sys.argv[3])
    
if __name__ == '__main__':
    main()

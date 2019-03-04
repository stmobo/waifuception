import pandas as pd
import numpy as np
from pathlib import Path
import sys

def main():
    base_path = Path('/mnt/data/dl/danbooru2018/original')
    
    have_ids = []
    fnames = {}
    jpg_fnames = {}
    for p in filter(lambda d: d.is_dir(), base_path.iterdir()):
        for f in filter(lambda p: p.is_file() and p.suffix in ['.png', '.jpg', '.jpeg'], p.iterdir()):
            iid = int(f.stem)
            have_ids.append(iid)
            fnames[iid] = f.parent.name + '/' + f.name
            jpg_fnames[iid] = f.parent.name + '/' + f.stem + '.jpg'
    
    print("Received {} images so far".format(len(have_ids)))
    
    df = pd.read_csv('./2018-full.csv.gz', index_col=None)    
    df_out = df[df['id'].isin(have_ids)]
    print(df_out.head())
    
    original_filenames = df_out.agg(lambda row: str(fnames[row['id']]), axis=1)
    preprocessed_filenames = df_out.agg(lambda row: str(jpg_fnames[row['id']]), axis=1)
    
    df_out.insert(2, 'original_filename', original_filenames)
    df_out.insert(2, 'filename', preprocessed_filenames)
    df_out.drop(['ext', 'image_width', 'image_height', 'source', 'character'], inplace=True, axis=1)
    print(df_out.head())
    
    print("Received {} training set images so far".format(len(df_out)))
    sys.stdout.flush()
    
    return

    df_out.to_csv('./2018-current.csv.gz', compression='gzip')
    
if __name__ == '__main__':
    main()
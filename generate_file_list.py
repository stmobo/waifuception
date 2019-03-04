import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./preprocessed/2018-full.csv.gz', index_col=0)
    
    with open('./dataset-images.txt', 'w', encoding='utf-8') as f:
        for i, row in enumerate(df.itertuples()):
            img_id = row[0]
            img_ext = row[1]
            folder_id = img_id % 1000
            
            path = "original/{:04d}/{:d}.{:s}".format(folder_id, img_id, img_ext)
            
            f.write(path+'\n')
            
            if (i+1) % 10000 == 0:
                print("Processed row "+str(i+1))
    
if __name__ == '__main__':
    main()

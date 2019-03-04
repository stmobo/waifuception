import pandas as pd
from pathlib import Path

def main():
    p = Path('./preprocessed')
    
    dfs = []
    
    for c in filter(lambda o: o.is_file() and o.suffix == '.csv', p.iterdir()):
        dfs.append(pd.read_csv(str(c), index_col=0))
        print("Read: {}".format(c.name))
        
    out_df = pd.concat(dfs)
    out_df.rename(columns={'1boy': 'male', '1girl': 'female'}, inplace=True)
    
    print(out_df.info())
    print(out_df.head())
    print(out_df.iloc[0])
    
    #out_df.to_hdf('./preprocessed/2018-full.h5', key='full_2018', complevel=9)
    out_df.to_csv('./preprocessed/2018-full.csv.gz', compression='gzip')
    
if __name__ == '__main__':
    main()

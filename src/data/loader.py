import pandas as pd
import os

def load_split(dataset, cfg):
    if dataset not in {'train', 'val', 'test'}:
        raise ValueError(f"Invalid split name: {dataset}")
    print ('im here')
    
    path = os.path.join('..', cfg['data']['processed_dir'], f"{dataset}.csv")
    df = pd.read_csv(path)

    texts = df['processed_text'].to_list()
    labels = df['sentiment']
    ids = df['doc_id']

    return texts, labels, ids




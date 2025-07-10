import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import ast

def load_dataset(path):
    df = pd.read_csv(path)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    return df

def encode_labels(df):
    label_encoder = LabelEncoder()
    all_labels = [label for sublist in df['labels'] for label in sublist]
    label_encoder.fit(all_labels)
    df['label_ids'] = df['labels'].apply(label_encoder.transform)

    return df, label_encoder

def get_hf_dataset(df):
    return Dataset.from_pandas(df[["tokens", "label_ids"]])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_contains_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    This function cleans the data by removing the rows with missing
    """
    for column in columns:
        df = df[df[column].notna()]

    return df

# 全ての要素が0である、不要な列を削除する関数
def remove_all_zeros(df):
    # すべてゼロのカラムを削除するための関数
    """全て0の列を削除"""
    df = df.copy()
    removed_cols = []
    for col in df.columns:
        if (df[col] == 0).all():
            removed_cols.append(col)
            df = df.drop(col, axis=1, inplace=True)

    return df

from elapy import elapy as ela
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
import torch
import os

def set_threshold(df: pd.DataFrame,
                  elements: list,
                  target: str,
                  target_values: list,
                  how: str = "mean") -> pd.DataFrame:

    # フィルタリング対象の行を抽出
    target_df = df[df[target].isin(target_values)]

    results = []
    for element in elements:
        if how == "mean":
            threshold_val = target_df[element].mean()
        elif how == "median":
            threshold_val = target_df[element].median()
        else:
            raise ValueError(f"Unsupported aggregation method: {how}")

        results.append((element, threshold_val))

    # 結果をDataFrame化して返す
    thresholds_df = pd.DataFrame(results, columns=["element", "threshold"])
    return thresholds_df

def data_binarization(df: pd.DataFrame, elements: list, threshold: list) -> pd.DataFrame:
    """
    This function binarizes the data
    """
    # 
    df = df.copy().loc[:, elements]

    for i, element in enumerate(elements):
        df[element] = df[element].apply(lambda x: 1 if x >= threshold[i] else 0)

    return df

def data_preprocessing(df_a: pd.DataFrame, df_b: pd.DataFrame, elements: list) -> pd.DataFrame:
    """
    This function preprocesses the data
    df_a: binarized data
    df_b: original data
    """
    df_a["class"] = 0
    for i in range(len(df_a)):
        for i, element in enumerate(elements):
            df_a.loc[i, "class"] += df_a.loc[i, element] * (2 ** (len(elements) - (i + 1)))
        
    df_b = pd.concat([df_b.reset_index(drop = True), df_a["class"]], axis=1)

    return df_a, df_b

def ela_fit(data : pd.DataFrame, elements: list) -> tuple:
    """
    This function fits the data to the ELA model bia elapy
    """
    # reform data
    data = data[elements].T
    data = data.astype(int)
    data.index = pd.MultiIndex.from_product([data.index, [""]], names=['', '0'])

    h, W = ela.fit_exact(data)

    # Check fitting accuracy scores (1: best, 0: worst).
    acc1, acc2 = ela.calc_accuracy(h, W, data)
    print(acc1, acc2)

    # Calculate a basin graph.
    graph = ela.calc_basin_graph(h, W, data)

    # Calculate a disconnectivity graph.
    D = ela.calc_discon_graph(h, W, data, graph)

    # Calculate each state's frequency and transitions between states.
    freq, trans, trans2 = ela.calc_trans(data, graph)

    # Calculate transition matrix based on Boltzmann machine
    P = ela.calc_trans_bm(h, W, data)

    freq = freq.reset_index()
    freq.rename(columns={"state_no":"index"})

    # Step 4: Plot results

    # Plot on-off patterns of each local minimum.
    ela.plot_local_min(data, graph)

    # Plot the basin graph.
    ela.plot_basin_graph(graph)

    # Plot the disconnectivity graph.
    if len(freq) != 1:
        ela.plot_discon_graph(D)

    # Plot each state's frequency and transitions between states.
    if len(freq) != 1:
        ela.plot_trans(freq, trans, trans2)

    return h, W, graph
import pandas as pd

# 現在のディレクトリを確認
import os
print(os.getcwd())
# 階層構造を確認
print(os.listdir())

# select your prepared data
df = df = pd.read_csv('input_data/example_data/cleaned_data/df_selected.csv', index_col=0, encoding='utf-8')

import MachineLearning as ML
df = ML.Datacleaning_for_ML(df)

# set feature columns
feature_columns = df.loc[:, :'g_Lachnoclostridium'].columns

params = ML.create_single_model_for_PT(df, 'nugent', feature_columns, n_trials = 10, seed = 0)
models, test_set_X, test_set_y = ML.create_k_models(df, 5, 'nugent', feature_columns, params = params, seed = 0)

ML.check_model_performance(models, test_set_X, test_set_y)

# SHAP
combines_shap_values = ML.cal_combined_shap(models, test_set_X)

# feature selection
selected_features = ML.choose_important_features(combines_shap_values, test_set_X, num_features = 5)

import ELA

thresholds = ELA.set_threshold(df, selected_features, "nugent" ,[0, 1, 2])
binarized_data = ELA.data_binarization(df, selected_features, thresholds["threshold"].to_list())

df_a, df_b = ELA.data_preprocessing(binarized_data, df, selected_features)

h,w,graph = ELA.ela_fit(df_a, selected_features)
# ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import shap
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor as LGBMR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from lightgbm import early_stopping, log_evaluation

# 全ての要素が0である、不要な列を削除する関数
def Datacleaning_for_ML(df):
    # すべてゼロのカラムを削除するための関数
    """全て0の列を削除"""
    df = df.copy()
    removed_cols = []
    for col in df.columns:
        if (df[col] == 0).all():
            removed_cols.append(col)
    df = df.drop(removed_cols, axis=1)
    
    return df

# パラメータチューニングを行う関数
def create_single_model_for_PT(df,target_column, feature_columns, n_trials = 500, seed = 0):
# モデリングはLightGBMを用い、SHAPを算出するためにsklearnのモデルを用いる
    X = df[feature_columns]
    y = df[target_column]

    # ログレベルを警告レベルに設定
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)   

    # パラメータチューニングを行う関数
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.02, 0.02),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'verbose': -1,
            'random_state': seed
        }

        model = LGBMR(**params)
        model.fit(X_train, y_train,eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)])
        y_pred = model.predict(X_test)
        # 0未満の予測値を0に、10以上の予測値を10に丸める
        y_pred = np.clip(y_pred, 0, 10)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        return rmse
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials,show_progress_bar=True)

    best_params = study.best_params
    model = LGBMR(**best_params)
    model.fit(X_train, y_train)

    # 予測値と実測値の比較
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.title('Predicted vs Actual' + " " + ' RMSE: ' + str(np.sqrt(np.mean((y_test - y_pred) ** 2))))
    plt.scatter(y_pred, y_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('output/figures/Model_performance/Model_performance_for_PT' + '.png')

    # best_paramsを返す
    return best_params

# データから複数のモデルを作成する関数(k_fold交差検証を行う)
def create_k_models(df, k ,target_column, feature_columns, k_fold_method="Stratified",params = "default", seed = 0):
    # モデリングはLightGBMを用い、SHAPを算出するためにsklearnのモデルを用いる
    X = df[feature_columns]
    y = df[target_column]

    if k_fold_method == "Stratified":
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    models = []
    test_set_X = []
    test_set_y = []

    if k_fold_method == "Stratified":
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # モデルの作成
            if params == "default":
                model = LGBMR(random_state=0, importance_type="gain", num_leaves=30, n_estimators=500, learning_rate=0.005, boosting_type="gbdt", objective="regression", metric="rmse")
            else:
                model = LGBMR(**params)
            model.fit(X_train, y_train)
            models.append(model)
            test_set_X.append(X_test)
            test_set_y.append(y_test)
    
    else:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # モデルの作成
            model = LGBMR(random_state=0, importance_type="gain", num_leaves=30, n_estimators=500, learning_rate=0.005, boosting_type="gbdt", objective="regression", metric="rmse")
            model.fit(X_train, y_train)
            models.append(model)
            test_set_X.append(X_test)
            test_set_y.append(y_test)

    return models, test_set_X, test_set_y

def check_model_performance(base_models, X, y):
    # モデルの評価を行う
    # R2スコア, RMSEを算出
    RMSE_list = []
    R2_list = []
    if type(base_models) == list:
        for i in range(len(base_models)):
            model = base_models[i]
            X_test = X[i]
            y_test = y[i]
            # テストデータの予測
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, 10)
            # RMSEの算出
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            R2 = model.score(X_test, y_test)
            RMSE_list.append(rmse)
            R2_list.append(R2)
            # テストデータの予測値と正解値の比較
            plt.figure(figsize=(8, 6))
            plt.title('Predicted vs Actual' + " " + ' RMSE: ' + str(rmse) + " " + ' R2: ' + str(R2))
            # 予測値と実測値をプロット
            plt.scatter(y_pred, y_test)
            # x = yの直線をプロット
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('output/figures/Model_performance/model' + str(i) + '.png')

    else:
        # テストデータの予測
        y_pred = model.predict(X)
        # テストデータの予測値と正解値の比較
        plt.figure(figsize=(8, 6))
        plt.title('Predicted vs Actual')
        plt.scatter(y_pred, y)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('output/figures/Model_performance/model' + str(i) + '.png')

    print("RMSE: ", RMSE_list)
    print("RMSE mean: ", np.mean(RMSE_list))
    print("R2: ", R2_list)
    print("R2 mean: ", np.mean(R2_list))

def cal_combined_shap(models, test_set_X):
    # すべてのモデルとテストデータを使ってSHAPを算出
    shap_values_all = []
    test_set_X_all = []

    # 各モデルのSHAP値を算出
    for i in range(len(models)):
        explainer = shap.TreeExplainer(models[i], test_set_X[i])
        shap_values = explainer(test_set_X[i])
        shap_values_all.append(shap_values)
        test_set_X_all.append(test_set_X[i])

    # 各モデルのshap_valuesの .values と .base_values を連結
    # SHAP値を統合
    test_set_X_all = pd.concat(test_set_X_all)  
    # 各shap_valuesオブジェクトの属性を結合
    concatenated_values = np.concatenate([sv.values for sv in shap_values_all], axis=0)
    concatenated_base_values = np.concatenate([sv.base_values for sv in shap_values_all], axis=0)
    concatenated_data = np.concatenate([sv.data for sv in shap_values_all], axis=0)

    # 統合したshap.Explanationオブジェクトを作成
    combined_shap_values = shap.Explanation(values=concatenated_values,
                                            base_values=concatenated_base_values,
                                            data=concatenated_data)

    return combined_shap_values
    

def choose_important_features(combined_shap_values, test_set_X_all, num_features = 5):
    # ① SHAP の数値配列を取り出す
    shap_array = combined_shap_values.values  # ExplanationオブジェクトからSHAP値の配列を取得

    # ② 多クラスの場合は shap_array.shape が (n_samples, n_classes, n_features) になることがある
    #    必要に応じてクラス平均や特定クラスのみを取り出すなどしてください。
    #    例: クラス 0 のみを取り出すなら shap_array = shap_array[:, 0, :]

    # ③ 各特徴量の平均絶対SHAP値を算出
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

    # ④ 特徴量名を取得（test_set_X_allがDataFrameの場合）
    features = test_set_X_all[0].columns

    # ⑤ DataFrame化して並べ替え
    summary_data = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    # ⑥ 上位 num_features 個の特徴量をリストで取得
    important_features = summary_data['feature'][:num_features].tolist()

    return important_features
# ベースイメージとしてMinicondaを使用
FROM continuumio/miniconda3

# 作業ディレクトリを指定
WORKDIR /app

# 必要なファイルをコンテナにコピー
COPY environment.yml /app/

# conda環境を作成
RUN conda env create -f environment.yml

# 必要なファイルをローカルからコンテナにコピー
COPY input_data /app/input_data
COPY output /app/output
COPY src /app/src

# conda環境を有効化するためにシェルを変更
SHELL ["conda", "run", "-n", "elaenv", "/bin/bash", "-c"]

# 実行するコマンドを指定
CMD ["python", "/app/src/execute.py", "--target_path", "/app/input_data", "--save_path", "/app/output"]


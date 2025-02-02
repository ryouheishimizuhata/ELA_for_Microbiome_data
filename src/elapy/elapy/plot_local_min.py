import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

def plot_local_min(data, graph):
    # 自己ループのデータをフィルタリング
    df = graph[graph.source == graph.target]
    
    # 状態が一つだけの場合の処理
    if df.state_no.nunique() == 1:
        print('Only one state')
        state = df.state_no.iloc[0]
        # 状態に対応する行だけを取得
        df = df[df.state_no == state]
        
        # バイナリ表現の配列を作成
        n = len(data)
        X = np.array([list(bin(i)[2:].rjust(n, '0'))
                       for i in df.index]).astype(int).T
        df = pd.DataFrame(X, index=data.index, columns=df.state_no)
    else:
        n = len(data)
        X = np.array([list(bin(i)[2:].rjust(n, '0'))
                       for i in df.index]).astype(int).T
        df = pd.DataFrame(X, index=data.index, columns=df.state_no)

    df = df.sort_index(axis=1)

    # ヒートマップの描画
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data=df, ax=ax, linecolor='w', lw=2, square=True,
                 cmap=sns.color_palette('Paired', 2),
                 cbar_kws=dict(ticks=[0.25, 0.75], shrink=0.25, aspect=2), vmax=1, vmin=0)
    
    ax.tick_params(length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Local minimum states', fontsize=16, pad=10)
    ax.set_xlabel('State number')
    ax.set_ylabel(None)
    
    # カラーバーの設定
    cax = ax.collections[0].colorbar.ax
    cax.set_yticklabels(['0', '1'])
    cax.tick_params(length=0)
    
    fig.tight_layout()
    plt.show()  # show()をここに移動
    fig.savefig('./output/figures/ela_results/fig_local_min.png')

if __name__ == '__main__':
  plot_local_min(data, graph)

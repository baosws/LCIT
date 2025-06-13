import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('')
from src.utils.metrics import scores

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

pd.set_option("display.precision", 2)

def viz(path, alpha):
    df = pd.read_csv(path)
    df = df[df.Z_cats != 0.5]
    df['Method'] = df['Method'].map(lambda x: x + ' (Ours)' if x == 'LCIT' else x)
    df['Pred'] = df['P_value'] >= alpha
    df[r'Types of $X$ and $Y$'] = 'Both discrete'
    df.loc[df.X_cat & (~df.Y_cat), r'Types of $X$ and $Y$'] = 'One discrete'
    df.loc[(~df.X_cat) & (~df.Y_cat), r'Types of $X$ and $Y$'] = 'Both continuous'
    # df['Z_cats'] = (df['Z_cats'] * 100).astype(int).astype(str) + '\%'
    id_vars = ['Linearity', 'N', 'd', 'X_cat', 'Y_cat', 'Z_cats', 'Method', r'Types of $X$ and $Y$']
    df = df.groupby(id_vars).apply(scores)
    print(df)

    score_names = df.columns
    df = pd.melt(df.reset_index(), id_vars=id_vars, value_vars=score_names, value_name='Value', var_name='Metric')
    df['Metric'] = df['Metric'].map({'F1': r'$F_1$ ($\uparrow$)', 'AUC': r'\textbf{AUC} ($\uparrow$)', 'Type I': r'\textbf{Type I error} ($\downarrow$)', 'Type II': r'\textbf{Type II error} ($\downarrow$)'})
    df[r'Type of $Z$'] = df['Z_cats'].map({0: 'All continuous', 1: 'All discrete'})
    g = sns.FacetGrid(df, col='Metric', sharey=True)
    g.map_dataframe(sns.lineplot, x='d', y='Value', hue=r'Type of $Z$', style=r'Types of $X$ and $Y$', markers=True, markersize=7, linewidth=2)
    g.set_xlabels(r'\textbf{Dimensionality}', fontsize=12)
    g.set_ylabels(r'Value (\%)', fontsize=12)
    g.set_titles(r'{col_name}', size=14)
    g.add_legend()
    for ax in g.axes.flat:
        ax.grid(axis='y')
    # g.fig.suptitle(rf'$\alpha = {alpha}$')
    g.savefig(path.replace('.csv', '.pdf'))
    return g

if __name__ == '__main__':
    viz('experiments/exp7/results/result.csv', alpha=0.05)
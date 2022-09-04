import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('')
from src.utils.metrics import scores

# matplotlib.rc('font', family='DejaVu Sans')
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['lines.linewidth'] = 1
# matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

pd.set_option("display.precision", 2)

def viz(path, alpha):
    df = pd.read_csv(path, index_col=0)
    df['Pred'] = df['P_value'] >= alpha
    id_vars = ['Linearity', 'd', 'Method']
    df = df.groupby(id_vars).apply(scores)
    print(df)

    score_names = df.columns
    df = pd.melt(df.reset_index(), id_vars=id_vars, value_vars=score_names, value_name='Value', var_name='Metric')
    g = sns.FacetGrid(df, row='Linearity', col='Metric', sharey=False)
    g.map_dataframe(sns.lineplot, x='d', y='Value', hue='Method', marker='o', markersize=7)
    g.set_xlabels(fontsize=10)
    g.set_ylabels(r'Value (%)', fontsize=10)
    g.set_titles('{row_name}–{col_name}', size=10)
    g.add_legend()
    for ax in g.axes.flat:
        ax.grid(axis='y')
    # g.fig.suptitle(rf'$\alpha = {alpha}$')
    g.savefig(path.replace('.csv', '.pdf'))
    return g

if __name__ == '__main__':
    viz('experiments/exp2/results/result.csv', alpha=0.05)
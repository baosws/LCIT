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
    id_vars = ['Dataset', 'Method']
    df = df.groupby(id_vars).apply(scores)[['AUC']]
    print(df)

    score_names = df.columns
    df = pd.melt(df.reset_index(), id_vars=id_vars, value_vars=score_names, value_name='Value', var_name='Metric')
    g = sns.FacetGrid(df, row='Dataset', col='Metric', sharey=False)
    g.map_dataframe(sns.barplot, x='Method', y='Value', hue='Method', palette='husl', dodge=False)
    # g.set(ylim=(50, None))
    g.set_xlabels(fontsize=10)
    g.set_ylabels(r'Value (%)', fontsize=10)
    g.set_titles('{row_name}–{col_name}', size=10)
    # g.add_legend()
    for ax in g.axes.flat:
        ax.grid(axis='y')
    # g.fig.suptitle(rf'$\alpha = {alpha}$')
    g.savefig(path.replace('.csv', '.pdf'))
    return g

if __name__ == '__main__':
    viz('experiments/exp5/results/result_D42.csv', alpha=0.05)
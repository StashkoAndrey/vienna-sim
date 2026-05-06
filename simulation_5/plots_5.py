# plots_5.py  ── consistent colour palette for A/B/C
import pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from parameters_extracted import (
    MAX_STEPS, REPLICATES,
    REFILL_INTERVAL_A, REFILL_INTERVAL_B, REFILL_INTERVAL_C,
    DELTA_T
)

# ───────────  global palette  ───────────
COL = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green'}

# ───────────  helpers  ───────────
def load_data(pkl='simulation_results.pkl'):
    with open(pkl, 'rb') as f:
        return pickle.load(f)

def pad(series, length):
    return np.pad(np.asarray(series, dtype=float), (0, length-len(series)), 'constant')

def pad_stack(list_of_series, length):
    return np.vstack([pad(s, length) for s in list_of_series])

def meal_lines(ax):
    # A meals – blue dashed
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        ax.axvline(step, color=COL['A'], ls='--', lw=.7,
                   label='A-meal' if step == REFILL_INTERVAL_A else '')
    # B meals – orange dotted
    step = DELTA_T
    while step < MAX_STEPS:
        ax.axvline(step, color=COL['B'], ls=':', lw=.7,
                   label='B-meal' if step == DELTA_T else '')
        step += REFILL_INTERVAL_B
    # C meals – green dash-dot
    step = 2*DELTA_T
    while step < MAX_STEPS:
        ax.axvline(step, color=COL['C'], ls='-.', lw=.7,
                   label='C-meal' if step == 2*DELTA_T else '')
        step += REFILL_INTERVAL_C


# ───────────  majority-by-channel population  ───────────
def plot_population_majority(d):
    A = pad_stack(d['pop_major_A'], MAX_STEPS)
    B = pad_stack(d['pop_major_B'], MAX_STEPS)
    C = pad_stack(d['pop_major_C'], MAX_STEPS)
    t = np.arange(MAX_STEPS)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, A.mean(0), label='A-majority', color=COL['A'])
    ax.plot(t, B.mean(0), label='B-majority', color=COL['B'])
    ax.plot(t, C.mean(0), label='C-majority', color=COL['C'])
    meal_lines(ax)

    ax.set(xlabel='Timestep', ylabel='Average # cells',
           title=f'Population classified by channel majority ({REPLICATES} replicates)')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig('avg_population_majority_ABC.png'); plt.close(fig)
    print('✔ saved → avg_population_majority_ABC.png')

# ───────────  population  ───────────
def plot_population(d):
    A = pad_stack(d['all_pop_A'], MAX_STEPS)
    B = pad_stack(d['all_pop_B'], MAX_STEPS)
    C = pad_stack(d['all_pop_C'], MAX_STEPS)
    t = np.arange(MAX_STEPS)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, A.mean(0), label='A-type', color=COL['A'])
    ax.plot(t, B.mean(0), label='B-type', color=COL['B'])
    ax.plot(t, C.mean(0), label='C-type', color=COL['C'])
    meal_lines(ax)

    ax.set(xlabel='Timestep', ylabel='Average # cells',
           title=f'Average population size ({REPLICATES} replicates)')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig('avg_population_ABC.png'); plt.close(fig)
    print('✔ saved → avg_population_ABC.png')

# ───────────  phage totals  ───────────
def plot_phage(d):
    phA = pad_stack(d['all_phage_A'], MAX_STEPS)
    phB = pad_stack(d['all_phage_B'], MAX_STEPS)
    phC = pad_stack(d['all_phage_C'], MAX_STEPS)
    t = np.arange(MAX_STEPS)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, phA.mean(0), label='Phage-A', color=COL['A'])
    ax.plot(t, phB.mean(0), label='Phage-B', color=COL['B'])
    ax.plot(t, phC.mean(0), label='Phage-C', color=COL['C'])
    meal_lines(ax)

    ax.set(xlabel='Timestep', ylabel='Avg # particles (log-scale)',
           title='Average phage counts')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig('avg_phage_ABC.png'); plt.close(fig)
    print('✔ saved → avg_phage_ABC.png')

# ───────────  energy  ───────────
def plot_energy(d):
    E = pad_stack(d['all_energy'], MAX_STEPS).mean(0)
    t = np.arange(MAX_STEPS)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, E, color='tab:purple', label='Total cellular energy')
    meal_lines(ax)

    ax.set(xlabel='Timestep', ylabel='Average energy (a.u.)',
           title='Average total energy in the grid')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig('avg_energy.png'); plt.close(fig)
    print('✔ saved → avg_energy.png')

# ───────────  channel-count heat-maps  ───────────
def channel_heatmap(series_by_rep, channel_name, cmap):
    agg = {}
    for rep in series_by_rep:
        for t, lst in rep.items():
            if lst: agg.setdefault(t, []).extend([c for c in lst if c > 0])
    if not agg:
        print(f'⚠ no data for {channel_name}')
        return

    times = sorted(agg)
    max_c = max(max(v) for v in agg.values())
    H = np.zeros((len(times), max_c + 1), int)
    for r, t in enumerate(times):
        for c in agg[t]:
            H[r, c] += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(H.T, ax=ax, cmap=cmap,
                cbar_kws={'label': 'Cell count'},
                xticklabels=times, yticklabels=range(1, max_c + 1))
    ax.set(xlabel='Timestep', ylabel=f'# {channel_name} channels',
           title=f'Distribution of {channel_name} over time')
    fig.tight_layout()
    fname = f'heat_{channel_name}.png'
    fig.savefig(fname); plt.close(fig)
    print(f'✔ saved → {fname}')

def plot_channel_heatmaps(d):
    channel_heatmap(d['all_chan_Atype'], 'A-channels (A-type cells)', 'Blues')
    channel_heatmap(d['all_chan_Btype'], 'B-channels (B-type cells)', 'Oranges')
    channel_heatmap(d['all_chan_Ctype'], 'C-channels (C-type cells)', 'Greens')

# ───────────  driver  ───────────
def main(pkl='simulation_results.pkl'):
    d = load_data(pkl)
    plot_population(d)
    plot_phage(d)
    plot_energy(d)
    plot_channel_heatmaps(d)
    plot_population_majority(d)

if __name__ == '__main__':
    main()

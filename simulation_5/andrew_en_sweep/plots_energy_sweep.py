# ======= plot_energy_sweep_fullmodel.py =======
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = 'energy_sweep_results.csv'

def plot_full_sweep(csv_file):
    df = pd.read_csv(csv_file)

    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    axs[0].plot(df['START_ENERGY'], df['AVG_BIOMASS'], marker='o', color='tab:green')
    axs[0].set_ylabel('Avg Biomass')
    axs[0].set_title('Average Biomass vs Start Energy')

    axs[1].plot(df['START_ENERGY'], df['AVG_ENERGY'], marker='o', color='tab:blue')
    axs[1].set_ylabel('Avg Energy')
    axs[1].set_title('Average Energy vs Start Energy')

    axs[2].plot(df['START_ENERGY'], df['AVG_LIFETIME'], marker='o', color='tab:purple')
    axs[2].set_ylabel('Avg Lifetime')
    axs[2].set_title('Average Lifetime vs Start Energy')

    axs[3].plot(df['START_ENERGY'], df['EXTINCTION_TIME'], marker='o', color='tab:red')
    axs[3].set_ylabel('Extinction Time')
    axs[3].set_xlabel('Start Energy')
    axs[3].set_title('Extinction Time vs Start Energy')

    plt.tight_layout()
    plt.savefig('energy_sweep_summary.png')
    print('✔ Saved → energy_sweep_summary.png')
    plt.show()

if __name__ == '__main__':
    plot_full_sweep(CSV_FILE)

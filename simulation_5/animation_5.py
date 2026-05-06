# animate_5.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from parameters_5 import GRID_SIZE

# ─────────── Setup ───────────
PKL = 'snapshot_run0.pkl'

with open(PKL, 'rb') as f:
    frames = pickle.load(f)

PHAGE_CMAP = {
    'A': plt.get_cmap('Blues'),
    'B': plt.get_cmap('Oranges'),
    'C': plt.get_cmap('Greens'),
}

# ─────────── Dashboard ───────────
def animate_dashboard(frames, save=True):
    num_frames = len(frames['channels'])

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Bacterial Channel Composition and Phage Spread", fontsize=14, weight='bold')

    ax_rgb, ax_pa, ax_pb, ax_pc = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    # RGB init
    rgb_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    im_rgb = ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Channel Composition")

    # Phage heatmaps
    ims = {}
    cbs = {}
    for ax, p in zip([ax_pa, ax_pb, ax_pc], ['A', 'B', 'C']):
        im = ax.imshow(frames[f'phage_{p}'][0], cmap=PHAGE_CMAP[p], vmin=0, vmax=10)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(f"Phage {p} Concentration")
        ax.set_title(f"Phage {p}")
        ims[p] = im
        cbs[p] = cb

    # Step counter overlay
    step_text = fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=12)

    def update(frame_idx):
        # RGB update
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                ch = frames['channels'][frame_idx][i][j]
                total = ch['A'] + ch['B'] + ch['C']
                if total > 0:
                    rgb_img[i, j] = np.array([ch['A'], ch['B'], ch['C']], dtype=float) / total
                else:
                    rgb_img[i, j] = 0
        im_rgb.set_data(rgb_img)

        # Phage updates
        for p in ['A', 'B', 'C']:
            ims[p].set_data(frames[f'phage_{p}'][frame_idx])

        step_text.set_text(f"Step: {frame_idx}")
        return [im_rgb, *ims.values(), step_text]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

    if save:
        ani.save(os.path.join('dashboard.gif'), writer='pillow', fps=5)
    else:
        plt.show()

# ─────────── Entry ───────────
if __name__ == '__main__':
    animate_dashboard(frames, save=True)

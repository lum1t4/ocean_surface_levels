import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xarray as xr


def generate_plot(src, dst):
    dataset = xr.load_dataset(src)
    SLA = dataset['sla']

    assert SLA.ndim == 3, "SLA data must be 3D (time, lat, lon)."

    # Compute global min/max for consistent colormap scaling
    vmin = float(SLA.min().values)
    vmax = float(SLA.max().values)

    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = None

    def update(frame):
        nonlocal cbar
        if cbar:
            cbar.remove()  # Clear the previous colorbar
        ax.clear()
        ax.set_title(f"SLA Heatmap - Time: {SLA.time[frame].values}")
        im = SLA.isel(time=frame).plot(
            ax=ax,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        return im,

    ani = FuncAnimation(fig, update, frames=len(SLA.time), interval=200, blit=False)
    ani.save(dst, writer="ffmpeg", fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SLA data as a video.")
    parser.add_argument("--src", type=str, help="Path to the SLA data file (NetCDF format).")
    parser.add_argument("--dst", type=str, help="Path to save the output video file.")
    args = parser.parse_args()
    generate_plot(args.src, args.dst)

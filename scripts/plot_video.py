import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import cmocean
import cartopy.crs as ccrs

def generate_plot(src, dst):
    # 1) Load dataset and pull SLA into memory (once)
    ds = xr.load_dataset(src)
    sla = ds['sla']
    assert sla.ndim == 3, "SLA data must be 3D (time, lat, lon)."
    sla = sla.load()  # now sla.values is in RAM

    # 2) Pre-compute global vmin/vmax
    vmin, vmax = float(sla.min()), float(sla.max())

    # 3) Setup figure & static map elements
    fig, ax = plt.subplots(
        figsize=(16, 9), dpi=150,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    # ax.set_extent([-30, -10, 22, 29.5], crs=ccrs.PlateCarree())
    coast = ax.coastlines(resolution='10m', color='gray', linewidth=1)
    ax.axis('off')

    # 4) Initial mesh (for frame 0)
    lon = sla.longitude
    lat = sla.latitude
    data0 = sla.isel(time=0).values

    mesh = ax.pcolormesh(
        lon, lat, data0,
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        cmap=cmocean.cm.balance
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05)
    title = ax.set_title(f"SLA Heatmap - Time: {np.datetime_as_string(sla.time[0].values, unit='D')}")

    # 5) Update function: only change the mesh array and title
    def update(frame):
        arr = sla.isel(time=frame).values.ravel()
        mesh.set_array(arr)
        t = np.datetime_as_string(sla.time[frame].values, unit='D')
        title.set_text(f"SLA Heatmap - Time: {t}")
        return mesh, title

    # 6) Animate with blit=True
    ani = FuncAnimation(
        fig, update,
        frames=sla.sizes['time'],
        interval=200,
        blit=True
    )

    # 7) Save with ffmpeg writer
    writer = FFMpegWriter(fps=10)
    ani.save(dst, writer=writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SLA data as a video.")
    parser.add_argument("--src", required=True, help="Path to SLA NetCDF file.")
    parser.add_argument("--dst", required=True, help="Output video path.")
    args = parser.parse_args()
    generate_plot(args.src, args.dst)

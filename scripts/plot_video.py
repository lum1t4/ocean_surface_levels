import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.animation import FFMpegWriter
from tqdm.auto import tqdm
from datetime import datetime

def main(src, dst):
    
    dataset = xr.open_dataset(src, chunks={"time": 1})
    sla = dataset["sla"]
    lon, lat = sla.longitude.values, sla.latitude.values
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
    ax.axis("off")


    vmin = np.nanmin(sla)
    vmax = np.nanmax(sla)

    mesh = ax.pcolormesh(lon, lat, np.zeros_like(sla[0]), vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, transform=ccrs.PlateCarree())
    title = ax.set_title("")
    plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.02)

    writer = FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"])

    with writer.saving(fig, dst, dpi=150):
        for t in tqdm(range(sla.sizes["time"]), desc="Rendering frames", unit="frame"):
            mesh.set_array(sla.isel(time=t).load().values.ravel())
            title.set_text(np.datetime_as_string(sla.time[t].values, unit="D"))
            writer.grab_frame()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", "--output", dest="dst", required=True)
    # Time range
    p.add_argument("--start_date", "-s", type=str, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end_date", "-e", type=str, help="End date (YYYY-MM-DD)")

    main(**vars(p.parse_args()))

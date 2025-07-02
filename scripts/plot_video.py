import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.animation import FFMpegWriter
from tqdm.auto import tqdm


def main(src, dst):
    bbox = dict(longitude=slice(-30.0625, -10), latitude=slice(22, 29.5))
    ds = xr.open_dataset(src, chunks={"time": 1})
    sla = ds["sla"].sel(**bbox).astype("float32").persist()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.set_extent([-30, -10, 22, 29.5])
    ax.axis("off")

    lon, lat = sla.longitude.values, sla.latitude.values
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.zeros_like(sla.isel(time=0)),
        vmin=float(sla.min()),
        vmax=float(sla.max()),
        cmap=cmocean.cm.balance,
        transform=ccrs.PlateCarree(),
    )
    title = ax.set_title("")
    plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.02)

    writer = FFMpegWriter(
        fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"]
    )

    with writer.saving(fig, dst, dpi=150):
        for t in tqdm(range(sla.sizes["time"]), desc="Rendering frames", unit="frame"):
            mesh.set_array(sla.isel(time=t).load().values.ravel())
            title.set_text(np.datetime_as_string(sla.time[t].values, unit="D"))
            writer.grab_frame()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    main(**vars(p.parse_args()))

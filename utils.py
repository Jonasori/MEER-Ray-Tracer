import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import imageio
from datetime import datetime


def gif_outputs(all_images, name=None):
    name = name if name else str(datetime.now()).replace(" ", "_").replace(":", "h")[:16]
    dpath = f"./plots/{name}/"
    if not os.path.exists(dpath):
        print("Making", dpath)
        os.makedirs(dpath)

    filenames = []
    for i, image in tqdm(enumerate(all_images), total=len(all_images)):
        fname = f"{dpath}{round(i/n * 100)}_pct.png"
        filenames.append(fname)

        plt.imshow(image)
        plt.savefig(fname)

    pngs = [imageio.imread(fname) for fname in tqdm(filenames)]
    imageio.mimsave(f"{dpath}total.gif", pngs)

    plt.clf()
    r_sum = [np.sum(im[:, :, 0]) for im in all_images]
    g_sum = [np.sum(im[:, :, 1]) for im in all_images]
    b_sum = [np.sum(im[:, :, 2]) for im in all_images]

    plt.plot(np.linspace(0, 12, n), r_sum, "-or")
    plt.plot(np.linspace(0, 12, n), b_sum, "-ob")
    plt.plot(np.linspace(0, 12, n), g_sum, "-og")

    plt.xlabel("Hour", weight="bold", fontsize=14)
    plt.ylabel("Intensity?", weight="bold", fontsize=14)
    plt.savefig(f"{dpath}intensity_timeseries.png")

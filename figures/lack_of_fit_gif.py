"""
Creates a gif to exemplify the lack of fit computation for a variable.

Author: Juan-Pablo Futalef
"""
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tqdm

target_file = "data/voi_losses/lack_of_fit/2024-05-09_15-16-30/GBM1_values.pkl"
reference_file = "data/voi_losses/lack_of_fit/2024-05-09_15-16-30/reference_values.pkl"

with open(target_file, "rb") as f:
    data = pickle.load(f)

with open(reference_file, "rb") as f:
    reference_data = pickle.load(f)

#%% Select data for the example
state_idx = 23

# Get the data
target_values = data[state_idx]
reference_values = reference_data[state_idx]

#%% ks statistic
import greyboxmodels.voi.metrics.lack_of_fit as lof

ks_data = {}
for tk, ref_val_array in reference_values.items():
    try:
        val_array = target_values[tk]
        ks_statistic, info = lof.ks_statistic(val_array, ref_val_array)
        ks_data[tk] = (ks_statistic, info)

    except KeyError:
        continue

#%% Create the gif

# Create the figure
plt.close("all")
fig, ax = plt.subplots(1, 3, figsize=(8, 4))

# Create the gif
images = []

current_t = []
current_ks = []
for t, (ks_value, info) in tqdm.tqdm(ks_data.items(), total=len(ks_data)):
    # PDF values
    bins = info["bins"]
    target_pdf = info["epdf1"]
    ref_pdf = info["epdf_ref"]

    # CDF values
    target_cdf = info["ecdf1"]
    ref_cdf = info["ecdf_ref"]

    # Plot the PDF using stairs
    ax[0].clear()
    ax[0].stairs(target_pdf, bins, label="Target", fill=True, alpha=0.5)
    ax[0].stairs(ref_pdf, bins, label="Reference", fill=True, alpha=0.5)
    ax[0].set_title("PDF")
    ax[0].legend(loc="upper right")

    # Plot the CDF
    ax[1].clear()
    cdf_x = np.linspace(target_pdf.min(), target_pdf.max(), len(target_cdf))
    ax[1].step(cdf_x, target_cdf, label="Target")
    ax[1].step(cdf_x, ref_cdf, label="Reference")
    ax[1].set_title("CDF")
    ax[1].legend(loc="lower right")
    ax[1].set_xlim([cdf_x.min(), cdf_x.max()])
    ax[1].set_ylim([-.1, 1.1])

    # Plot the ks statistic
    ax[2].clear()
    current_t.append(t)
    current_ks.append(ks_value)
    ax[2].plot(current_t, current_ks, color="black")
    ax[2].set_title("KS statistic")

    ks_val_mean = np.mean(current_ks)
    ax[2].axhline(ks_val_mean, color="black", linestyle="--")

    # Annotation
    # ax[2].annotate(f"Mean: {ks_val_mean:.2f}", (current_t[-1], ks_val_mean), textcoords="offset points", xytext=(10, 0), ha='center')

    ax[2].set_xlim([0, current_t[-1]])

    # Figure title with time
    fig.suptitle(f"Time: {t:.2f} s")

    # Tight layout
    fig.tight_layout()

    # Save the figure
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

#%% import dill as pickle

# with open("figures/lof_gif_images.pkl", "wb") as f:
#     pickle.dump(images, f)

#%% Open
import dill as pickle
with open("figures/lof_gif_images.pkl", "rb") as f:
    images = pickle.load(f)

#%% Save the gif
imageio.mimsave("figures/lof.gif", images, duration=0.5)


#%% to mp4 video
import cv2

# Get the shape of the images
height, width, layers = images[0].shape

# Create the video
video = cv2.VideoWriter("figures/lof.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 7, (width, height))

# Write the images
for image in images:
    video.write(image)

# Release the video
video.release()







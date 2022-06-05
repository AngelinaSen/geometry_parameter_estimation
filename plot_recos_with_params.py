from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import odl
import os
import scipy.io
import logging
import argparse

# Block of constants
DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm
SOURCE_RADIUS = 859.46  # distance between the source and COR
N_PROJ = 360  # number of projection angles

N_IMG = 1000  # image resolution
DOMAIN_L = 200
CUT = 500
RECO_SPACE = odl.uniform_discr(
    min_pt=[-DOMAIN_L, -DOMAIN_L], max_pt=[DOMAIN_L, DOMAIN_L], shape=[N_IMG, N_IMG], dtype="float32"
)
DETECTOR_DEV_PARTITION = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)

PARAM_PATH = "params/"
LOG_PHANTOM_FN = "./log_phantom/phantom_sino.mat"


def load_sinogram_data(file_name: str) -> np.ndarray:
    """ Function to load the sinogram data
    :param file_name: name of the file containing sinogram
    :return calibration_disk: matrix containing the sinogram data
    """
    sino = scipy.io.loadmat(file_name)
    sino = sino["arr"]
    sino[:, CUT:DETECTOR_LENGTH_PX] = 0
    return sino


def gather_params(in_dir: str, out_fn: str) -> np.ndarray:
    """
    FIXME:
    """
    data_files = sorted([file for file in os.listdir(in_dir) if not file.startswith(".")])
    # print(data_files)
    param_matrix = []
    for i in data_files:
        # print(i)
        params = np.load(in_dir + i)
        # print(params)
        param_matrix.append(params)

    params_save = np.array(param_matrix)
    # print(params_save.shape)
    np.save(out_fn, params_save)
    logging.info(f"The set of optimal parameters is saved to: {out_fn}")
    return params_save


def geom_with_par(par: Tuple) -> odl.tomo.geometry.conebeam.FanBeamGeometry:
    """ Function to set the measurement geometry
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :return geometry_device: measurement geometry
    """
    first_angle = par[0]
    src_r = SOURCE_RADIUS
    det_r = par[1]  # DETECTOR_RADIUS
    src_shift = par[2]  # par[1]
    det_shift = par[3]  # par[2]
    ax_shift = par[4]  # par[3]
    angle_dev_partition = odl.uniform_partition(first_angle, first_angle + 2 * np.pi, N_PROJ)
    geometry_device = odl.tomo.geometry.conebeam.FanBeamGeometry(
        angle_dev_partition,
        DETECTOR_DEV_PARTITION,
        src_radius=src_r,
        det_shift_func=lambda angle: np.array([[0, det_shift]]),
        src_shift_func=lambda angle: np.array([[0, src_shift]]),
        det_radius=det_r,
        det_axis_init=[1, ax_shift],
    )

    return geometry_device


def reco_with_par(par: Tuple, sino: np.ndarray) -> np.ndarray:
    """ Function to produce FBP-reconstruction
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :param sino: sinogram data
    :return reco: FBP-reconstruction
    """
    geometry_device = geom_with_par(par)
    ray_device = odl.tomo.RayTransform(RECO_SPACE, geometry_device, impl="astra_cpu")
    fbp = odl.tomo.fbp_op(ray_device, filter_type="Hann", frequency_scaling=1.0)
    return fbp(sino).data


def plot_recos_with_diff_params(params: np.ndarray, sinogram: np.ndarray, output_img_fn: str) -> None:
    """ Function to make a set of subplots - reconstructions with different parametrisations
    :params: array of different parametrisations
    :sinogram: sinogram data from which the reconstructions will be obtained
    :output_img_fn: name of a file where the plot will be saved
    """
    n_col = 3
    n_row = 2

    fig, axs = plt.subplots(n_row, n_col, constrained_layout=False, figsize=(18, 11))
    # fig.tight_layout()
    fig.add_gridspec(n_row, n_col, wspace=0, hspace=0)

    images = []

    for i in range(n_row):
        for j in range(n_col):
            par = params[i * n_col + j, :]

            reco = reco_with_par(par, sinogram)

            images.append(axs[i, j].imshow(reco, cmap="gray"))
            axs[i, j].axis("off")

            axs[i, j].set_title(
                rf"$\alpha_0=$ {par[0]:.2f}, $r_D=$ {par[1]:.2f}"
                + "\n"
                + rf"$h_S=$ {par[2]:.2f}, $h_D=$ {par[3]:.2f},"
                + "\n"
                + rf"$\alpha_D=$ {par[4]:.2f}",
                fontsize=16,
            )

    # Find the min and max of all colors for use in setting the color scale
    v_min = min(image.get_array().min() for image in images)  # 0.0
    v_max = max(image.get_array().max() for image in images)  # 0.02
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs, orientation="vertical", fraction=0.1)
    tick_font_size = 16
    cbar.ax.tick_params(labelsize=tick_font_size)

    def update(changed_image):
        for im in images:
            if changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim():
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect("changed", update)

    plt.savefig(output_img_fn)


def main():
    # logging settings
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
    )

    # arguments
    parser = argparse.ArgumentParser(description="Process command line " "arguments")

    parser.add_argument(
        "--disk-dir",
        "-d",
        dest="disk_dir",
        default="./hole_disk/",
        help="Path to the calibration phantom (ground truth + simulated X-ray data)",
    )

    args = parser.parse_args()
    logging.info(f"Directory with the calibration phantom: {args.disk_dir}")

    # get geometry parameters
    param_dir = args.disk_dir + PARAM_PATH
    out_fn = args.disk_dir + "geometry_parameters.npy"
    param_array = gather_params(param_dir, out_fn)  # get array of optimal parameters

    # make plots
    log_sino = load_sinogram_data(LOG_PHANTOM_FN)  # sinogram of a log phantom
    output_img_fn = args.disk_dir + "recos_with_params.png"
    plot_recos_with_diff_params(param_array, log_sino, output_img_fn)

    logging.info(f"The plot is saved to: {output_img_fn}")


if __name__ == "__main__":
    main()

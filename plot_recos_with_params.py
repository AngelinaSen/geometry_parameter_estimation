from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import odl
import os
import scipy.io
import logging
import argparse

from utils.quality_metrics import calculate_rel_error
from utils.data_parser import load_sinogram_data, CalibrationDisk

# to make LaTeX-style plots
import matplotlib
matplotlib.rcParams.update({"font.size": 14})  # 30 for chains, 18 for solution
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True

# Block of constants
DPI = 300  # for plots
DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm
SOURCE_RADIUS = 859.46  # distance between the source and COR
N_PROJ = 360  # number of projection angles

N_DISCR = 256  # image resolution
DOMAIN_L = 200
CUT = 500
RECO_SPACE = odl.uniform_discr(
    min_pt=[-DOMAIN_L, -DOMAIN_L], max_pt=[DOMAIN_L, DOMAIN_L], shape=[N_DISCR, N_DISCR], dtype="float32"
)
DETECTOR_DEV_PARTITION = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)

PARAM_PATH = "params/"
LOG_PHANTOM_FN = "./log_phantom/phantom_sinogram_noise_2.npy"


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
            if i == 0 and j == 0:
                # reference image (true geometry parameters)
                par = [2.55, 714.68, 319.94, 43.65, 0.28]
            else:
                par = params[(i-1) * n_col + (j-1), :]

            # make reconstruction with geometry parameters
            reco = reco_with_par(par, sinogram)

            # store the reconstruction
            images.append(axs[i, j].imshow(reco, cmap="gray"))

            # set plot labels
            if i == 0 and j == 0:  # if reference image (reconstruction with true geometry parameters)
                ref_img = reco
                axs[i, j].set_title("True parameters:" + "\n" +
                                    rf"$\alpha_0={par[0]:.2f}$, $r_D={par[1]:.2f}$" + "\n" +
                                    rf"$h_S= {par[2]:.2f}$, $h_D={par[3]:.2f}$," + "\n" +
                                    rf"$\alpha_D={par[4]:.2f}$", fontweight="bold", fontsize=23)

            else:  # if reconstructions with other optimal parametrisations
                # compute relative error
                relerr_metric = calculate_rel_error(ref_img, reco)

                axs[i, j].set_title(rf"$\alpha_0={par[0]:.2f}$, $r_D={par[1]:.2f}$" + "\n" +
                                    rf"$h_S= {par[2]:.2f}$, $h_D={par[3]:.2f}$," + "\n" +
                                    rf"$\alpha_D={par[4]:.2f}$", fontsize=23)

                axs[i, j].set_ylabel(r"$\varepsilon_{\mathrm{rel}} = $ "
                                     f"{relerr_metric:.3f}", fontsize=24)
                # axs[i, j].label_outer()

            # to get rid of ticks and tick labels
            plt.setp(axs[i, j].get_xticklabels(), visible=False)
            plt.setp(axs[i, j].get_yticklabels(), visible=False)
            axs[i, j].tick_params(axis='both', which='both', length=0)

    # set the color scale
    v_min = 0
    v_max = 1
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs, orientation="vertical", fraction=0.1)
    tick_font_size = 28
    cbar.ax.tick_params(labelsize=tick_font_size)

    def update(changed_image):
        for im in images:
            if changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim():
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect("changed", update)

    plt.savefig(output_img_fn, dpi=DPI)


def main():
    # logging settings
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
    )

    # arguments
    parser = argparse.ArgumentParser(description="Process command line " "arguments")

    parser.add_argument(
        "--disk",
        "-d",
        dest="calibration_disk",
        choices=[
            CalibrationDisk.L_DISK.value,
            CalibrationDisk.HOLE_DISK.value,
        ],
        default=CalibrationDisk.L_DISK,
        type=CalibrationDisk,
        help="Calibration phantom (L-shaped disk or disk with a hole)",
    )

    parser.add_argument('--n-proj', '-p', dest='n_proj',
                        choices=[360, 180, 90, 45, 20], default=20, type=int,
                        help='Number of projection angles used in the geometry parameter search')

    args = parser.parse_args()

    # TODO: add error handling if the calibration disk is wrong
    if args.calibration_disk == CalibrationDisk.L_DISK:
        disk_dir = "./L_disk/"

    elif args.calibration_disk == CalibrationDisk.HOLE_DISK:
        disk_dir = "./hole_disk/"
    else:
        print("ERROR: wrong calibration disk")
        return

    # get geometry parameters
    # load optimal parameters from the file
    output_dir = disk_dir + PARAM_PATH
    param_fn = output_dir + f"params_{args.calibration_disk.value}_{args.n_proj}_ang.npy"
    param_array = np.load(param_fn)

    # make plots
    log_sino = np.load(LOG_PHANTOM_FN)  # sinogram of a log phantom
    output_img_fn = args.disk_dir + "recos_with_params.png"
    plot_recos_with_diff_params(param_array, log_sino, output_img_fn)

    logging.info(f"The plot is saved to: {output_img_fn}")


if __name__ == "__main__":
    main()

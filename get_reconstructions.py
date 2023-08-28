import os
import numpy as np
import odl
import logging
import argparse
import h5py
import subprocess

import scipy
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.io import savemat

import matplotlib.pyplot as plt
from matplotlib import colors

from utils.quality_metrics import calculate_rel_error
from utils.data_parser import extract_angles_from_sino, CalibrationDisk

# to make LaTeX-style plots
import matplotlib
matplotlib.rcParams.update({"font.size": 20})  # 30 for chains, 18 for solution
matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
matplotlib.rcParams["text.usetex"] = True

# geometry parameters
DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm
SOURCE_RADIUS = 859.46  # distance between the source and COR

# geometry constants used in Julia code
SRC_SHIFT_A1 = 0.0
DET_SHIFT_A1 = 0.0
SRC_TO_DET_INIT_A1 = 0.0
SRC_TO_DET_INIT_A2 = 1.0
DET_AXIS_INIT_A1 = 1.0

# plot resolution
DPI = 300

# space parameters (uniform discretization)
N_DISCR = 256
HALF_SIDE = 200
# plot parameters
N_ROWS = 6
N_COL = 3

CUT = 500

# Filename constants
PARAM_PATH = "params/"
LOG_PHANTOM_FN = "./log_phantom/phantom_sinogram_noise_2.npy"


def ray_transform(par: np.ndarray, n_angles: int) -> odl.tomo.RayTransform:
    """
    Forward operator
    :param par: geometry parameter vector
    :param n_angles: number of projection angles
    :return: forward operator
    """

    first_angle = par[0]
    src_r = SOURCE_RADIUS
    det_r = par[1]  # DETECTOR_RADIUS
    src_shift = par[2]  # par[1]
    det_shift = par[3]  # par[2]
    ax_shift = par[4]  # par[3]

    space = odl.uniform_discr([-HALF_SIDE, -HALF_SIDE], [HALF_SIDE, HALF_SIDE], [N_DISCR, N_DISCR], dtype="float32")

    angle_partition = odl.uniform_partition(first_angle, first_angle + 2 * np.pi, n_angles)

    detector_partition = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)

    # Geometry for the log with calibration disk
    geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SOURCE_RADIUS,
        src_shift_func=lambda x: np.array([SRC_SHIFT_A1, src_shift], dtype=float, ndmin=2),
        det_radius=det_r,
        det_shift_func=lambda angle: [DET_SHIFT_A1, det_shift],
        det_axis_init=[DET_AXIS_INIT_A1, ax_shift],
    )

    return odl.tomo.RayTransform(space, geometry, impl="astra_cpu")


def get_fbp_reco(sino: np.ndarray, ray_transf: odl.tomo.RayTransform) -> np.ndarray:
    """
    Computes FBP-reconstruction.
    :param sino: sinogram
    :param ray_transf: forward operator
    :return: FBP-reconstruction
    """

    fbp = odl.tomo.fbp_op(ray_transf, filter_type="Hann", frequency_scaling=1.0)
    ray_transf.geometry.src_shift_func(ray_transf.geometry.angles)
    return fbp(sino)


def get_tikhonov_reco(sino: np.ndarray, ray_transf: odl.tomo.RayTransform) -> np.ndarray:
    """ Computes reconstruction using Tikhonov regularization.
    :param sino: sinogram
    :param ray_transf: forward operator
    :return: Tikhonov-reconstruction
    """

    # Tikhonov
    space = odl.uniform_discr([-HALF_SIDE, -HALF_SIDE], [HALF_SIDE, HALF_SIDE], [N_DISCR, N_DISCR], dtype="float32")
    B = odl.IdentityOperator(space)
    a = 1

    T = ray_transf.adjoint * ray_transf + a * B.adjoint * B
    b = ray_transf.adjoint(sino)

    f = space.zero()
    odl.solvers.conjugate_gradient(T, f, b, niter=10)

    return f


def get_isocauchy_reco(
    sino: np.ndarray,
    ray_operator: odl.tomo.RayTransform,
    par: np.ndarray,
    ang: int,
    use_julia_matrix: bool = False
) -> np.ndarray:
    """
    Computes MAP estimates with Cauchy priors
    :param sino: sinogram
    :param ray_operator: forward operator
    :param par: geometry parameter vector
    :param ang: number of projection angles
    :param use_julia_matrix: boolean parameter that controls usage of the system matrix created in Julia
    :return data: reconstruction
    """

    map_par = 0.01
    likeli_var = 0.3 ** 2
    BFGS_iter = 150

    identifier = str(N_DISCR) + "x" + str(ang)
    datafile_folder = "matrices/"
    if not os.path.exists(datafile_folder):
        os.makedirs(datafile_folder)

    sino_file = datafile_folder + "sinog.mat"
    savemat(sino_file, {"sino": sino})  # Save sinogram in this file for Julia
    map_fname = datafile_folder + identifier + "_map_estimate.mat"  # Julia will save the MAP estimate in this file

    base_command = [
        "julia", "-t", "5", "-O1", "theorymatrix.jl",
        "--N", str(N_DISCR),
        "--NRAYS", str(DETECTOR_LENGTH_PX),
        "--NPROJ", str(ang),
        "--SIDE_2", str(HALF_SIDE),
        "--DETECTOR_LENGTH_MM", str(DETECTOR_LENGTH_MM),
        "--DETECTOR_RADIUS", str(par[1]),
        "--SOURCE_RADIUS", str(SOURCE_RADIUS),
        "--SRC_TO_DET_INIT_A1", str(SRC_TO_DET_INIT_A1),
        "--SRC_TO_DET_INIT_A2", str(SRC_TO_DET_INIT_A2),
        "--DET_AXIS_INIT_A1", str(DET_AXIS_INIT_A1),
        "--DET_AXIS_INIT_A2", str(par[4]),
        "--DET_SHIFT_A1", str(DET_SHIFT_A1),
        "--DET_SHIFT_A2", str(par[3]),
        "--SRC_SHIFT_A1", str(SRC_SHIFT_A1),
        "--SRC_SHIFT_A2", str(par[2]),
        "--SINO_FILE", str(sino_file),
        "--COLUMNMAJOR", "false",
        "--MAP_FILE", str(map_fname),
        "--MAP_PAR", str(map_par),
        "--LIKELI_VAR", str(likeli_var),
        "--INITANGLE", str(par[0]),
    ]
    if use_julia_matrix:
        # Julia will save its theory matrix in the file
        matrix_fname = datafile_folder + identifier + "_juliamatrix.mat"
        base_command.extend(
            [
                "--MATRIX_FILE", str(matrix_fname),
                "--MAP", "true",
                "--OTHER_MATRIX", "false",
                "--BFGS_ITER", str(BFGS_iter),
            ]
        )
        subprocess.run(base_command)
        f = h5py.File(map_fname, "r")
        return np.array(f.get("/map/"))
    else:
        radon_operator = sp.csc_matrix(create_system_matrix(ray_operator, N_DISCR))
        odl_matrix_fname = (
            datafile_folder + identifier + "_odlmatrix" + ".mat"
        )  # Save ODL theory matrix in this file for Julia
        savemat(odl_matrix_fname, {"radonmatrix": radon_operator})
        base_command.extend(
            [
                "--OTHER_MATRIX_FILE", str(odl_matrix_fname),
                "--MAP", "true",
                "--OTHER_MATRIX", "true",
                "--BFGS_ITER", str(BFGS_iter),
            ]
        )
        subprocess.run(base_command)
        f = h5py.File(map_fname, "r")
        return np.array(f.get("/map/"))


def plot_set_of_recos(sinogram_data: np.ndarray, par: np.ndarray, out_fn: str) -> None:
    """
    Plots different types of reconstructions (FBP, Tikhonov, MAP estimates with Cauchy prior)
    for different number of projection angles
    :param sinogram_data: the sinogram data (full-angle)
    :param par: geometry parameter vector
    :param out_fn: name  of  a file to save the plot
    :return:
    """
    angles = [360, 180, 90, 45, 20]
    n_angs = len(angles)
    recos = []

    n_methods = 3

    fig, axs = plt.subplots(n_methods, n_angs, figsize=(24, 12))
    fig.add_gridspec(n_angs, n_methods, wspace=0, hspace=0)

    images = []

    logging.info("Obtain reconstructions with different methods for various number of projection angles: ")

    for i in range(n_methods):
        for j in range(n_angs):
            ang = angles[j]
            logging.info(f"Method - {i+1}, number of projection angles - {ang}")
            # leave only needed number of projection angle in sinogram
            sinogram = extract_angles_from_sino(sinogram_data, ang)
            # create forward map, ray operator
            ray_operator = ray_transform(par, ang)
            if i == 0:
                # get FBP-reconstruction
                reco = get_fbp_reco(sinogram, ray_operator)
                ylabel = "FBP"
            elif i == 1:
                # get Tikhonov reconstruction
                reco = get_tikhonov_reco(sinogram, ray_operator)
                ylabel = "Tikhonov"
            elif i == 2:
                # get Cauchy reconstruction using theory matrix
                if (ang < 21):
                    usejuliamatrix = False
                else:
                    usejuliamatrix = True
                reco = get_isocauchy_reco(sinogram, ray_operator, par, ang, use_julia_matrix=usejuliamatrix)
                ylabel = "MAP"
            else:
                raise ValueError("Wrong number of reconstruction methods: should be less or equal than 3")

            if i == 0 and j == 0:
                true_img = reco
            else:
                # compute the metrics
                relerr_metric = calculate_rel_error(true_img, reco)

            recos.append(reco)

            images.append(axs[i, j].imshow(reco, cmap='gray'))
            axs[i, j].set_ylabel(ylabel, fontsize=33)
            axs[i, j].label_outer()

            if i == 0 and j == 0:
                axs[i, j].xaxis.set_label_position('top')
                axs[i, j].set_xlabel("Reference image", fontsize=24)
            else:
                axs[i, j].xaxis.set_label_position('top')
                # axs[i, j].set_xlabel(f"SSIM $={ssim_metric:.3f} $", fontsize=30)
                axs[i, j].set_xlabel(r"$\varepsilon_{\mathrm{rel}} = $ "
                                     f"{relerr_metric:.3f}", fontsize=23)

            if i == 0:
                axs[i, j].set_title(str(ang) + " angles", fontsize=33)

            # to get rid of ticks and tick labels
            plt.setp(axs[i, j].get_xticklabels(), visible=False)
            plt.setp(axs[i, j].get_yticklabels(), visible=False)
            axs[i, j].tick_params(axis='both', which='both', length=0)

    # Set the color scale
    v_min = 0
    v_max = 1
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    for im in images:
        im.set_norm(norm)

    # colorbar font size
    cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(30)

    def update(changed_image):
        for im in images:
            if changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim():
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect("changed", update)

    plt.savefig(out_fn, dpi=DPI)
    plt.show()


def create_system_matrix(ray_op: odl.tomo.RayTransform, n_discr: int) -> scipy.sparse.csr_matrix:
    """
    Plots different types of reconstructions (FBP, Tikhonov, MAP estimates with Cauchy prior)
    for different number of projection angles
    :param ray_op: forward operator
    :param n_discr: number of elements in the discretization
    :return:
    """
    # get system matrix
    sys_mat = odl.matrix_representation(ray_op)
    system_matrix = np.reshape(sys_mat, (ray_op.range.shape[0] * DETECTOR_LENGTH_PX, n_discr ** 2))
    return scipy.sparse.csr_matrix(system_matrix)


def main():
    # logging settings:
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

    # load sinogram of the digital log phantom
    log_sino = np.load(LOG_PHANTOM_FN)

    # load optimal parameters from the file
    output_dir = disk_dir + PARAM_PATH
    param_fn = output_dir + f"params_{args.calibration_disk.value}_{args.n_proj}_ang.npy"
    opt_params = np.load(param_fn)[0, :-1]

    # Get FBP, Tikhonov, MAP with Cauchy reconstructions and plot them
    plot_set_of_recos(log_sino, opt_params, f"reconstructions_{args.calibration_disk.value}_{args.n_proj}_ang.png")

    print("DONE")


if __name__ == "__main__":
    main()

import os
import numpy as np
import odl
import logging
import h5py
import subprocess

import scipy
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.io import savemat

import matplotlib.pyplot as plt
from matplotlib import colors

# geometry parameters
DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm

SRC_TO_DET_INIT_A1 = 0.0
SRC_TO_DET_INIT_A2 = 1.0
DET_AXIS_INIT_A1 = 1.0
DET_AXIS_INIT_A2 = 2.80799451e-01
DET_SHIFT_A1 = 0.0
DET_SHIFT_A2 = 4.36514375e01
SRC_SHIFT_A1 = 0.0
SRC_SHIFT_A2 = 3.19943788e02
SOURCE_RADIUS = 859.46  # distance between the source and COR
DETECTOR_RADIUS = 7.14683839e02  # SOURCE_DETECTOR_DIST - SOURCE_RADIUS
INITANGLE = 2.55023549e00

# parameters for angle partition
N_ANGLES = 360
END_ANGLE = 2 * np.pi + INITANGLE

# space parameters (uniform discretization)
N_DISCR = 128
SIDE_2 = 200
# plot parameters
N_ROWS = 6
N_COL = 3

CUT = 500

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


def extract_angles_from_sino(sino: np.ndarray, angle_num: int) -> np.ndarray:
    """ Extracts projection data from the full-angle sinogram
    for a given number of evenly located projection angles

    :param sino: sinogram
    :param angle_num: number of projection angles
    :return: sinogram containing data for given angles only
    """
    step = 360 // angle_num
    return sino[0:360:step, :]


def ray_transform(
    half_side: int,
    n_discr: int,
    n_angles: int,
    start_angle: float,
    end_angle: int
) -> odl.tomo.RayTransform:
    """ Forward operator
    :param half_side: half-side of the space discretization domain
    :param n_discr: number of elements in space discretization
    :param n_angles: number of angles for the angle discretization
    :param start_angle: the first angle for discretization
    :param end_angle: the last angle for discretization
    :return: forward operator
    """
    space = odl.uniform_discr([-half_side, -half_side], [half_side, half_side], [n_discr, n_discr], dtype="float32")

    angle_partition = odl.uniform_partition(start_angle, end_angle, n_angles)

    detector_partition = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)

    # Geometry for the log with calibration disk
    geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=SOURCE_RADIUS,
        src_shift_func=lambda x: np.array([SRC_SHIFT_A1, SRC_SHIFT_A2], dtype=float, ndmin=2),
        det_radius=DETECTOR_RADIUS,
        det_shift_func=lambda angle: [DET_SHIFT_A1, DET_SHIFT_A2],
        det_axis_init=[DET_AXIS_INIT_A1, DET_AXIS_INIT_A2],
    )

    return odl.tomo.RayTransform(space, geometry, impl="astra_cpu")


def get_fbp_reco(sino: np.ndarray, ray_transf: odl.tomo.RayTransform) -> np.ndarray:
    """ Computes FBP-reconstruction.

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
    space = odl.uniform_discr([-SIDE_2, -SIDE_2], [SIDE_2, SIDE_2], [N_DISCR, N_DISCR], dtype="float32")
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
    ang: int,
    use_julia_matrix: bool = False
) -> np.ndarray:
    """ Computes MAP estimates with Cauchy priors

    :param sino: sinogram
    :param ray_operator: forward operator
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
        "--SIDE_2", str(SIDE_2),
        "--DETECTOR_LENGTH_MM", str(DETECTOR_LENGTH_MM),
        "--DETECTOR_RADIUS", str(DETECTOR_RADIUS),
        "--SOURCE_RADIUS", str(SOURCE_RADIUS),
        "--SRC_TO_DET_INIT_A1", str(SRC_TO_DET_INIT_A1),
        "--SRC_TO_DET_INIT_A2", str(SRC_TO_DET_INIT_A2),
        "--DET_AXIS_INIT_A1", str(DET_AXIS_INIT_A1),
        "--DET_AXIS_INIT_A2", str(DET_AXIS_INIT_A2),
        "--DET_SHIFT_A1", str(DET_SHIFT_A1),
        "--DET_SHIFT_A2", str(DET_SHIFT_A2),
        "--SRC_SHIFT_A1", str(SRC_SHIFT_A1),
        "--SRC_SHIFT_A2", str(SRC_SHIFT_A2),
        "--SINO_FILE", str(sino_file),
        "--COLUMNMAJOR", "false",
        "--MAP_FILE", str(map_fname),
        "--MAP_PAR", str(map_par),
        "--LIKELI_VAR", str(likeli_var),
        "--INITANGLE", str(INITANGLE),
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


def plot_set_of_recos(sinogram_data: np.ndarray, out_fn: str) -> None:
    """ Plots different types of reconstructions (FBP, Tikhonov, MAP estimates with Cauchy prior)
    for different number of projection angles
    :param sinogram_data: the sinogram data (full-angle)
    :param out_fn: name  of  a file to save the plot
    :return:
    """
    angles = [360, 180, 90, 45, 20]
    n_angs = len(angles)
    recos = []

    fig, axs = plt.subplots(N_COL, n_angs, constrained_layout=False)
    # fig.tight_layout()
    fig.suptitle("FBP-reconstructions, Tikhonov reconstructions, MAPs with Cauchy prior")
    fig.add_gridspec(n_angs, N_COL, wspace=0, hspace=0)

    images = []

    logging.info("Obtain reconstructions with different methods for various number of projection angles: ")

    for i in range(N_COL):
        for j in range(n_angs):
            ang = angles[j]
            logging.info(f"Method - {i+1}, number of projection angles - {ang}")
            # leave only needed number of projection angle in sinogram
            sinogram = extract_angles_from_sino(sinogram_data, ang)
            # create forward map, ray operator
            ray_operator = ray_transform(SIDE_2, N_DISCR, ang, INITANGLE, END_ANGLE)
            if i == 0:
                # get FBP-reconstruction
                reco = get_fbp_reco(sinogram, ray_operator)
                ylabel = "FBP"
            elif i == 1:
                # get Tikhonov reconstruction
                reco = get_tikhonov_reco(sinogram, ray_operator)
                ylabel = "Tikhonov"
            elif i == 2:
                reco = get_isocauchy_reco(sinogram, ray_operator, ang)
                ylabel = "MAP"
            else:
                raise ValueError("N_COL should be less or equal than 3")

            recos.append(reco)
            images.append(axs[i, j].imshow(reco, cmap="gray"))
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].label_outer()
            if i == 0:
                axs[i, j].set_title(str(ang) + " angles")

    # Find the min and max of all colors for use in setting the color scale
    v_min = min(image.get_array().min() for image in images)
    v_max = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation="vertical", fraction=0.1)

    def update(changed_image):
        for im in images:
            if changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim():
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect("changed", update)

    plt.savefig(out_fn)
    plt.show()


def create_system_matrix(ray_op: odl.tomo.RayTransform, n_discr: int) -> scipy.sparse.csr_matrix:
    """ Plots different types of reconstructions (FBP, Tikhonov, MAP estimates with Cauchy prior)
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

    log_sino = load_sinogram_data(LOG_PHANTOM_FN)  # sinogram of a log phantom

    # Get FBP, Tikhonov, MAP with Cauchy reconstructions and plot them
    plot_set_of_recos(log_sino, "reconstructions.png")

    print("DONE")


if __name__ == "__main__":
    main()

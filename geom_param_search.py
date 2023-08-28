from typing import Tuple, Any

import numpy as np
import odl


from scipy import optimize

from datetime import datetime
import os

import logging
import argparse

from utils.data_parser import load_sinogram_data, extract_angles_from_sino, load_calibration_disk, CalibrationDisk

# Block of constants:

# Filename constants
PARAM_PATH = "params/"

# Geometry related constants
N_PROJ = 360  # number of projection angles (full angle tomography)
N_IMG = 256  # image resolution
N_PARAMS = 5  # number of parameters to optimise

DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm

SOURCE_RADIUS = 859.46  # distance between the source and COR (known parameter)

DOMAIN_L = 200

RECO_SPACE = odl.uniform_discr(
    min_pt=[-DOMAIN_L, -DOMAIN_L], max_pt=[DOMAIN_L, DOMAIN_L], shape=[N_IMG, N_IMG], dtype="float32"
)
DETECTOR_DEV_PARTITION = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)


def geom_with_par(par: Tuple[float], n_proj: int) -> odl.tomo.geometry.conebeam.FanBeamGeometry:
    """
    Function to set the measurement geometry
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :param n_proj: number of projection angles
    :return geometry_device: measurement geometry
    """
    first_angle = par[0]
    src_r = SOURCE_RADIUS
    det_r = par[1]
    src_shift = par[2]
    det_shift = par[3]
    ax_shift = par[4]
    angle_dev_partition = odl.uniform_partition(first_angle, first_angle + 2 * np.pi, n_proj)
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


def reco_with_par(par: Tuple[float], sino: np.ndarray) -> np.ndarray:
    """ Function to produce FBP-reconstruction
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :param sino: sinogram data
    :return reco: FBP-reconstruction
    """
    # compute number of projection angles
    n_proj = sino.shape[0]

    geometry_device = geom_with_par(par, n_proj)
    ray_device = odl.tomo.RayTransform(RECO_SPACE, geometry_device, impl="astra_cpu")
    fbp = odl.tomo.fbp_op(ray_device, filter_type="Hann", frequency_scaling=1.0)  # Ram-Lak filter
    return fbp(sino).data


def objective_function(par: Tuple[float], *args: Any) -> float:
    """
    Objective function, used in the optimisation procedure (Differential Evolution) to find a set
    of geometry parameters that produce nicely looking FBP-reconstructions
    :param par: geometry parameters - the initial angle, the source radius, the detector radius, the source shift,
                the detector shift, the detector tilt
    :param args: tuple of additional parameters - sinogram of the calibration disk,
                images of the calibration disk and flipped calibration disk
    :return v: objective function
    """
    sino = args[0]
    object_compa = args[1]
    object_compa_2 = args[2]
    fbp_reconstruction = reco_with_par(par, sino)
    corr_1 = -np.sum(fbp_reconstruction * object_compa)
    corr_2 = -np.sum(fbp_reconstruction * object_compa_2)
    v = min(corr_1, corr_2)

    logging.info(f"parameters: '{par}', object function value: '{v}'")

    return v


def estimate_parameters(sino: np.ndarray, gr_truth_img: np.ndarray, gr_truth_img_flipped: np.ndarray) -> np.ndarray:
    """
    Function to run Differential Evolution optimisation to estimate geometry parameters
    :param sino: sinogram of the calibration phantom
    :param gr_truth_img: image of the calibration phantom
    :param gr_truth_img_flipped: mirrored image of the calibration phantom
    :return parameters: vector of optimal parameters + value of cost function J(theta)
    """

    bnds = ((0, 2 * np.pi), (500, 1000), (-500, 500), (-500, 500), (-1, 1))  # bounds for optimization

    res = optimize.differential_evolution(objective_function, bnds, (sino, gr_truth_img, gr_truth_img_flipped))

    # save geometry parameter vector and also values of J(theta)
    res_vec = np.append(res.x, [res.fun], axis=0)
    return res_vec


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

    parser.add_argument('--n-runs', '-n', dest='n_runs', default=5, type=int,
                        help='Number of optimal parameter vectors to obtain (= number of program runs)',)

    args = parser.parse_args()

    # TODO: add error handling if the calibration disk is wrong
    if args.calibration_disk == CalibrationDisk.L_DISK:
        disk_dir = "./L_disk/"

    elif args.calibration_disk == CalibrationDisk.HOLE_DISK:
        disk_dir = "./hole_disk/"
    else:
        print("ERROR: wrong calibration disk")
        return

    # load calibration disk (ground truth + simulated X-ray data)
    disk = load_calibration_disk(disk_dir + "data/disk.png", N_IMG)  # calibration phantom
    disk_flipped = load_calibration_disk(disk_dir + "data/disk_flipped.png", N_IMG)  # mirrored calibration phantom

    disk_sino = load_sinogram_data(disk_dir + "data/sino.mat")
    if args.n_proj < 360:
        disk_sino = extract_angles_from_sino(disk_sino, args.n_proj)  # extract angles is needed

    # estimate geometry parameters
    logging.info(f"Geometry parameter estimation with {args.calibration_disk.value} ({args.n_proj} projection angles) "
                 f"is in progress...")

    optimal_params = np.empty((args.n_runs, N_PARAMS + 1))
    for i in range(args.n_runs):
        print(f"Iteration {i}")
        # run geometry parameter search
        optimal_params[i, :] = estimate_parameters(disk_sino, disk, disk_flipped)

    # create the output directory for output parameters (if it does not exist)
    output_dir = disk_dir + PARAM_PATH
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # file name to save parameters
    param_fn = output_dir + f"params_{args.calibration_disk.value}_{args.n_proj}_ang.npy"

    # save result to a file
    np.save(param_fn, optimal_params)
    logging.info(f"Optimal parameters are saved to: {param_fn}")


if __name__ == "__main__":
    main()

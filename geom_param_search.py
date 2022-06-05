from typing import Tuple, Any

import numpy as np
import odl

from scipy import optimize
from PIL import Image
import scipy.io
from datetime import datetime
import os

import logging
import argparse


# Block of constants:

# Filename constants
PARAM_PATH = "params/"

# Geometry related constants
N_PROJ = 360  # number of projection angles (full angle tomography)
N_IMG = 1000  # image resolution

DETECTOR_LENGTH_PX = 768  # length of detector in pixels
DETECTOR_LENGTH_MM = 1154.2  # length of detector in mm

SOURCE_RADIUS = 859.46  # distance between the source and COR
SOURCE_DETECTOR_DIST = 1491.28  # distance between the source and the corresponding detector ( 1506 )
DETECTOR_RADIUS = SOURCE_DETECTOR_DIST - SOURCE_RADIUS

CUT = 500  # used to load sinogram data
DOMAIN_L = 200

RECO_SPACE = odl.uniform_discr(
    min_pt=[-DOMAIN_L, -DOMAIN_L], max_pt=[DOMAIN_L, DOMAIN_L], shape=[N_IMG, N_IMG], dtype="float32"
)
DETECTOR_DEV_PARTITION = odl.uniform_partition(-DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_MM / 2, DETECTOR_LENGTH_PX)


def load_sinogram_data(file_name: str) -> np.ndarray:
    """ Function to load the sinogram data
    :param file_name: name of the file containing sinogram
    :return calibration_disk: matrix containing the sinogram data
    """
    sino = scipy.io.loadmat(file_name)
    sino = sino["arr"]
    sino[:, CUT:DETECTOR_LENGTH_PX] = 0
    return sino


def load_calibration_disk(file_name: str) -> np.ndarray:
    """ Function to load the calibration disk
    :param file_name: name of the file containing a calibration disk
    :return calibration_disk: matrix containing the calibration disk representation
    """
    mat = Image.open(file_name).convert("L")  # L-shaped calibration disk
    wpercent = N_IMG / float(mat.size[0])
    size = int((float(mat.size[1]) * float(wpercent)))
    mat = mat.resize((size, size), Image.NEAREST)
    return np.array(mat) / 255.0


def geom_with_par(par: Tuple[float]) -> odl.tomo.geometry.conebeam.FanBeamGeometry:
    """ Function to set the measurement geometry
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :return geometry_device: measurement geometry
    """
    first_angle = par[0]
    src_r = SOURCE_RADIUS
    det_r = par[1]
    src_shift = par[2]
    det_shift = par[3]
    ax_shift = par[4]
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


def reco_with_par(par: Tuple[float], sino: np.ndarray) -> np.ndarray:
    """ Function to produce FBP-reconstruction
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :param sino: sinogram data
    :return reco: FBP-reconstruction
    """
    geometry_device = geom_with_par(par)
    ray_device = odl.tomo.RayTransform(RECO_SPACE, geometry_device, impl="astra_cpu")
    fbp = odl.tomo.fbp_op(ray_device, filter_type="Hann", frequency_scaling=1.0)  # Ram-Lak filter
    return fbp(sino).data


def objective_function(par: Tuple[float], *args: Any) -> float:
    """ Objective function, used in optimization procedure (Differential Evolution) to find a set
    of geometry parameters that produce nicely looking FBP-reconstructions
    :param par: geometry parameters - starting angle, source radius, detector radius, source shift, detector shift,
                detector tilt
    :param args: tuple of additional parameters - sinogram, calibration disk and flipped calibration disk
    :return v: objective function
    """
    sino = args[0]
    object_compa = args[1]
    object_compa_2 = args[2]
    fbp_reconstruction = reco_with_par(par, sino)
    corr1 = -np.sum(fbp_reconstruction * object_compa)
    corr2 = -np.sum(fbp_reconstruction * object_compa_2)
    v = min(corr1, corr2)

    logging.info(f"parameters: '{par}', object function value: '{v}'")

    return v


def estimate_parameters(sino: np.ndarray, gr_truth_img: np.ndarray, gr_truth_img_flipped: np.ndarray) -> np.ndarray:
    """ Function to run differential evolution optimization to estimate geometry parameters
    :param sino: sinogram of a calibration phantom
    :param gr_truth_img: image of a calibration phantom
    :param gr_truth_img_flipped: mirrored image of a calibration phantom
    :return parameters: vector of optimal parameters
    """
    # Bounds for optimization
    bnds = ((0, 2 * np.pi), (500, 1000), (-500, 500), (-500, 500), (-1, 1))

    res = optimize.differential_evolution(objective_function, bnds, (sino, gr_truth_img, gr_truth_img_flipped))
    return res.x


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

    parser.add_argument("--id-num", "-i", dest="id", default=1, type=int, help="Process ID number")

    args = parser.parse_args()
    logging.info(f"Directory with the calibration phantom: {args.disk_dir}")

    # load calibration disk (ground truth + simulated X-ray data)
    disk = load_calibration_disk(args.disk_dir + "data/disk.png")  # calibration phantom
    disk_flipped = load_calibration_disk(args.disk_dir + "data/disk_flipped.png")  # mirrored calibration phantom
    disk_sino = load_sinogram_data(args.disk_dir + "data/sino.mat")

    # estimate geometry parameters
    logging.info("Geometry parameter estimation is in progress...")
    par = estimate_parameters(disk_sino, disk, disk_flipped)

    # create the output directory for output parameters (if it does not exist)
    output_dir = args.disk_dir + PARAM_PATH
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save result to a file
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    out_fn = output_dir + f"parameters_{date}"
    np.save(out_fn, par)
    logging.info(f"Optimal parameters are saved to: {out_fn}")


if __name__ == "__main__":
    main()

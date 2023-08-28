import numpy as np
import scipy.io
import enum
from PIL import Image


# constants
CUT = 500  # used to load sinogram data
DETECTOR_LENGTH_PX = 768  # length of detector in pixels


class CalibrationDisk(str, enum.Enum):
    L_DISK = "L-disk"
    HOLE_DISK = "hole-disk"


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
    """
    Extracts projection data from the full-angle sinogram
    for a given number of evenly located projection angles
    :param sino: sinogram
    :param angle_num: number of projection angles
    :return: sinogram containing data for given angles only
    """

    step = 360 // angle_num
    sinogram = sino[0:360:step, :]
    return sinogram


def load_calibration_disk(file_name: str, n_discr: int) -> np.ndarray:
    """ Function to load the calibration disk
    :param file_name: name of the file containing a calibration disk
    param n_discr: number of discretisation points (image size)
    :return calibration_disk: matrix containing the calibration disk representation
    """
    mat = Image.open(file_name).convert("L")  # L-shaped calibration disk
    wpercent = n_discr / float(mat.size[0])
    size = int((float(mat.size[1]) * float(wpercent)))
    mat = mat.resize((size, size), Image.NEAREST)
    return np.array(mat) / 255.0


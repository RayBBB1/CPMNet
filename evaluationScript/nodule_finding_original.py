import math
from typing import List
import numpy as np
# from .nodule_typer import NoduleTyper
IMAGE_SPACING = [1, 0.8, 0.8]

def to_float(x):
    return float(x) if x is not None else None

def compute_sphere_volume(diameter: float) -> float:
    if diameter == 0:
        return 0
    elif diameter == -1:
        return 100000000
    else:
        radius = diameter / 2
        return 4/3 * math.pi * radius**3

def compute_nodule_volume(w: float, h: float, d: float) -> float:
    # We assume that the shape of the nodule is approximately spherical. The original formula for calculating its 
    # volume is 4/3 * math.pi * (w/2 * h/2 * d/2) = 4/3 * math.pi * (w * h * d) / 8. However, when comparing the 
    # segmentation volume with the nodule volume, we discovered that using 4/3 * math.pi * (w * h * d) / 8 results 
    # in a nodule volume smaller than the segmentation volume. Therefore, we opted to use 4/3 * math.pi * (w * h * d) / 6 
    # to calculate the nodule volume.
    volume = 4/3 * math.pi * ((w * h * d) / 6)
    return volume

NODULE_TYPE_DIAMETERS = {'benign': [0,4],
                        'probably_benign': [4, 6],
                        'probably_suspicious': [6, 8],
                        'suspicious': [8, -1]}

class NoduleTyper:
    def __init__(self, 
                 image_spacing: List[float]):
        self.nodule_type_diameters = NODULE_TYPE_DIAMETERS
        self.spacing = np.array(image_spacing, dtype=np.float64)
        self.voxel_volume = np.prod(self.spacing)
        
        self.areas = {}
        for key in self.nodule_type_diameters:
            self.areas[key] = [round(compute_sphere_volume(self.nodule_type_diameters[key][0]) / self.voxel_volume),
                               round(compute_sphere_volume(self.nodule_type_diameters[key][1]) / self.voxel_volume)]
        
    def get_nodule_type_by_seg_size(self, nodule_size: float) -> str:
        for key in self.areas:
            if nodule_size >= self.areas[key][0] and (nodule_size < self.areas[key][1] or self.areas[key][1] == -1):
                return key
        return 'benign'
    
    def get_nodule_type_by_dhw(self, d: int, h: int, w: int) -> str:
        nodule_volume = compute_nodule_volume(w, h, d)
        return self.get_nodule_type_by_seg_size(nodule_volume)

class NoduleFinding:
    """
    Represents a nodule
    """
    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
            CADprobability=None, nodule_type=None, w=None, h=None, d=None, seriesInstanceUID=None):

        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = to_float(coordX)
        self.coordY = to_float(coordY)
        self.coordZ = to_float(coordZ)
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.w = to_float(w)
        self.h = to_float(h)
        self.d = to_float(d)
        self.candidateID = None
        self.seriesuid = seriesInstanceUID
        self.nodule_type = nodule_type
        
    def auto_nodule_type(self):
        nodule_typer = NoduleTyper(IMAGE_SPACING)
        self.nodule_type = nodule_typer.get_nodule_type_by_dhw(self.d, self.h, self.w)
        
    def __str__(self) -> str:
        return 'NoduleFinding: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(self.coordX, self.coordY, self.coordZ, self.w, self.h, self.d)
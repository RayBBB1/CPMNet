import math
from typing import List
import numpy as np
from .nodule_typer_old import NoduleTyper
IMAGE_SPACING = [1, 0.8, 0.8]

def to_float(x):
    if x is None:
        return None
    else:
        return float(x)

class NoduleFinding(object):
    '''
    Represents a nodule
    '''
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
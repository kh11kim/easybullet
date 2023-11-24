from __future__ import annotations
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import numpy as np
from typing import *
from typing_extensions import override

Array = np.ndarray
ArrayLike = Union[np.ndarray, tuple, list]

class SO3(Rotation):
    def __init__(self, xyzw):
        super().__init__(quat=xyzw, normalize=True)
    
    @override
    def __repr__(self) -> str:
        xyzw = np.round(self.xyzw, 5)
        return f"{self.__class__.__name__}(xyzw={xyzw})"
    
    def __matmul__(self, target: Union[SO3, np.ndarray]):
        if isinstance(target, SO3):
            return self.multiply(target)
        elif isinstance(target, np.ndarray):
            return self.apply(target)
        raise ValueError
    
    @property
    def xyzw(self): return self.as_quat()
    @property
    def wxyz(self): return np.roll(self.xyzw, 1)
    
    # factory
    @classmethod
    def random(cls):
        xyzw = Rotation.random().as_quat()
        return cls(xyzw)
    @classmethod
    def identity(cls): return cls([0,0,0,1])
    @classmethod
    def from_quat(cls, xyzw): return SO3(xyzw)
    @classmethod
    def from_rotvec(cls, rotvec): 
        return cls.from_Rotation(Rotation.from_rotvec(rotvec))
        
    @classmethod
    def from_Rotation(cls, rot:Rotation): return cls(rot.as_quat())
    @classmethod
    def from_wxyz(cls, wxyz): return cls(np.roll(wxyz, -1))
    @override
    @classmethod
    def from_euler(cls, seq: str, angles: Sequence[float]):
        """{‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations"""
        return cls.from_Rotation(Rotation.from_euler(seq, angles))
    @override
    @classmethod
    def from_matrix(cls, mat:ArrayLike):
        return cls.from_Rotation(Rotation.from_matrix(mat))
    @classmethod
    def exp(cls, tangent):
        return cls.from_Rotation(Rotation.from_rotvec(tangent))
    
    def inverse(self):
        return SO3(self.xyzw * np.array([-1, -1, -1, 1]))
    @override
    def apply(self, target: ArrayLike) -> ArrayLike:
        return super().apply(target)
    def multiply(self, other: SO3):
        x0, y0, z0, w0 = self.xyzw
        x1, y1, z1, w1 = other.xyzw
        xyzw=np.array([
            x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
            -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
            x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
        ])
        return SO3(xyzw)
    def log(self):
        return self.as_rotvec()


class SE3:
    def __init__(self, rot:SO3 = SO3.identity(), trans:ArrayLike=np.zeros(3)):
        assert isinstance(rot, SO3)
        assert isinstance(trans, (np.ndarray, list, tuple))
        self.rot = rot
        self.trans = np.asarray(trans)
    
    def __repr__(self) -> str:
        xyz = np.round(self.trans, 5)
        xyzw = np.round(self.rot.xyzw, 5)
        return f"{self.__class__.__name__}(xyzw={xyzw}, xyz={xyz})"
    
    @property
    def xyzw(self):
        return self.rot.xyzw

    #factory
    @classmethod
    def from_matrix(cls, mat:Array):
        assert isinstance(mat, np.ndarray) and mat.shape == (4, 4)
        rot = SO3.from_matrix(mat[:3,:3])
        trans = mat[:3, -1]
        return cls(rot, trans)
    @classmethod
    def from_xyz_xyzw(cls, xyz_xyzw):
        xyz, xyzw = xyz_xyzw[:3], xyz_xyzw[-4:]
        return cls(rot=SO3(xyzw), trans=np.asarray(xyz))
    @classmethod
    def random(cls, lower=-np.ones(3), upper=np.ones(3)):
        rot = SO3.random()
        trans = np.random.uniform(lower, upper)
        return cls(rot, trans)
    
    def as_matrix(self):
        return np.vstack(
            (np.c_[self.rot.as_matrix(), self.trans], [0.0, 0.0, 0.0, 1.0])
        )
    def as_xyz_xyzw(self):
        return np.hstack([self.trans, self.rot.xyzw])
    
    def multiply(self, other:SE3):
        rotation = self.rot @ other.rot
        translation = self.rot.apply(other.trans) + self.trans
        return SE3(rotation, translation)
    
    def apply(self, target:Array):
        assert target.shape == (3,) or target.shape[1] == 3
        return self.rot.apply(target) + self.trans
    
    def inverse(self):
        rot = self.rot.inverse()
        trans = -rot.apply(self.trans)
        return SE3(rot, trans)
    
    @classmethod
    def identity(cls):
        return cls()

    def __matmul__(self, target):
        if isinstance(target, SE3):
            return self.multiply(target)
        elif isinstance(target, np.ndarray):
            return self.apply(target)
        raise ValueError
    


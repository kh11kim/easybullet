import abc
from typing import *
import numpy as np
import pybullet as p
from attrs import define, field
from .pose import SE3, SO3


@define(frozen=True)
class Shape(abc.ABC):
    rgba: Tuple
    ghost: bool
    viz_offset_xyz_xyzw: Tuple[float]
    col_offset_xyz_xyzw: Tuple[float]

    def get_viz_query(self) -> dict:
        return dict(
            visualFramePosition=self.viz_offset.trans,
            visualFrameOrientation=self.viz_offset.rot.xyzw,
            rgbaColor=self.rgba
        )
    
    def get_col_query(self):
        return dict(
            collisionFramePosition=self.col_offset.trans,
            collisionFrameOrientation=self.col_offset.rot.xyzw,
        )
    
    @property
    def viz_offset(self):
        return SE3.from_xyz_xyzw(self.viz_offset_xyz_xyzw)
    @property
    def col_offset(self):
        return SE3.from_xyz_xyzw(self.col_offset_xyz_xyzw)


@define(frozen=True)
class CylinderShape(Shape):
    radius: float
    length: float

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.length,
        )
        return query
    
    def get_col_query(self):
        query = super().get_col_query()
        query.update(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.length,
        )
        return query

@define(frozen=True)
class MeshShape(Shape):
    viz_mesh_path: str
    col_mesh_path: str
    scale: float
    centering: bool

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_MESH,
            fileName=self.viz_mesh_path,
            meshScale=np.ones(3)*self.scale,
        )
        return query
    
    def get_col_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_MESH,
            fileName=self.col_mesh_path,
            meshScale=np.ones(3)*self.scale,
        )
        return query
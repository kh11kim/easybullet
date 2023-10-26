from __future__ import annotations

import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from .pose import SE3, SO3
from dataclasses import dataclass, field
import trimesh
from typing import *
import abc
import time


Array = np.ndarray
ArrayLike = Union[np.ndarray,List,Tuple]


@dataclass(frozen=True)
class Shape(abc.ABC):
    rgba: tuple
    ghost: bool
    viz_offset_xyzwxyz: Tuple[float]
    col_offset_xyzwxyz: Tuple[float]

    def get_viz_query(self) -> dict:
        return dict(
            visualFramePosition=self.viz_offset_xyzwxyz[-3:],
            visualFrameOrientation=self.viz_offset_xyzwxyz[:4],
            rgbaColor=self.rgba
        )
    
    def get_col_query(self):
        return dict(
            collisionFramePosition=self.viz_offset_xyzwxyz[-3:],
            collisionFrameOrientation=self.viz_offset_xyzwxyz[:4],
        )

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class MeshShape(Shape):
    viz_mesh_path: str
    col_mesh_path: str
    scale: float

    def get_viz_query(self):
        return dict(
            shapeType=p.GEOM_MESH,
            fileName=self.viz_mesh_path,
            meshScale=np.ones(3)*self.scale,
            rgbaColor=self.rgba,
            visualFramePosition=self.viz_offset_xyzwxyz[-3:],
            visualFrameOrientation=self.viz_offset_xyzwxyz[:4])
    
    def get_col_query(self):
        return dict(
            shapeType=p.GEOM_MESH,
            fileName=self.col_mesh_path,
            meshScale=np.ones(3)*self.scale,
            collisionFramePosition=self.col_offset_xyzwxyz[-3:],
            collisionFrameOrientation=self.col_offset_xyzwxyz[:4])

class BulletWorld(BulletClient):
    def __init__(self, gui=True):
        connection_mode = p.GUI if gui else p.DIRECT
        super().__init__(connection_mode=connection_mode)
        if gui:
            self.pause_button_uid = p.addUserDebugParameter("pause",1,0,1)
        self.shapes = {}
        self.body_dict: Dict[Body] = {}
    
    def register_body(self, body:Body):
        self.body_dict[body.name] = body
        #print(f"multibody registered: {body.name}[{body.uid}]")

    def remove_body(self, body:Union[Body, Bodies]):
        if isinstance(body, Body):
            if body.name in self.body_dict:    
                self.removeBody(body.uid)
                del self.body_dict[body.name]
                #print(f"multibody removed: {body.name}[{body.uid}]")
        elif isinstance(body, Bodies):
            for body_ in body.bodies:
                self.remove_body(body_)
    
    def step(self, no_dynamics=False):
        if no_dynamics:
            self.performCollisionDetection()
        else:
            self.stepSimulation()

    def get_shape_id(self, shape:Shape):
        if shape in self.shapes:
            return self.shapes[shape]
        
        viz_id = self.createVisualShape(**shape.get_viz_query())
        col_id = -1 if not shape.ghost \
            else self.createCollisionShape(**shape.get_col_query())
        
        self.shapes[shape] = (viz_id, col_id)
        return self.shapes[shape]
    
    def show(self):
        """function for macos"""
        num_quit = p.readUserDebugParameter(self.pause_button_uid)
        
        dt = 0.01
        while True:
            self.step(no_dynamics=True)
            time.sleep(dt)
            quit = p.readUserDebugParameter(self.pause_button_uid)
            if quit >= num_quit+1: break
        

@dataclass
class Body(abc.ABC):
    name: str
    shape: Shape
    mass: float
    client: BulletWorld
    uid: int = field(init=False)
    viz_id: int = field(init=False)
    col_id: int = field(init=False)

    def __post_init__(self):
        assert self.name not in self.client.body_dict, "Body name already exists!"
        self.viz_id, self.col_id = self.client.get_shape_id(self.shape)
        self.uid = self.client.createMultiBody(
            baseVisualShapeIndex=self.viz_id,
            baseCollisionShapeIndex=self.col_id,
            baseMass=self.mass)
        self.client.register_body(self)

    def set_pose(self, pose: SE3):
        self.client.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat())
    
    def get_pose(self):
        pos, orn = self.client.getBasePositionAndOrientation(self.uid)
        return SE3(SO3(orn), pos)

    @abc.abstractclassmethod
    def get_shape(cls):
        pass

@dataclass
class Mesh(Body):
    shape: MeshShape

    @classmethod
    def get_shape(
            cls, viz_mesh_path:str, col_mesh_path=None, 
            rgba=None, scale=1., centering=False,
            viz_offset=None, col_offset=None):
        ghost = True if col_mesh_path is None else False
        if centering:
            mesh = trimesh.load(viz_mesh_path)
            viz_offset = SE3(trans=-mesh.centroid)
            col_offset = SE3(trans=-mesh.centroid)
        viz_offset = viz_offset if viz_offset is not None else SE3.identity()
        col_offset = col_offset if col_offset is not None else SE3.identity()
        rgba = None if rgba is None else tuple(rgba)
        
        shape = MeshShape(
            viz_mesh_path=viz_mesh_path,
            col_mesh_path=col_mesh_path,
            scale=scale,
            viz_offset_xyzwxyz=tuple(viz_offset.as_xyz_xyzw()),
            col_offset_xyzwxyz=tuple(col_offset.as_xyz_xyzw()),
            rgba=rgba,
            ghost=ghost,
        )
        return shape

@dataclass
class Cylinder(Body):
    shape: CylinderShape

    @classmethod
    def get_shape(cls, radius, length, rgba=(1,1,1,1), ghost=False, offset: SE3 = None):
        offset = SE3.identity() if offset is None else offset
        offset = tuple(offset.as_xyz_xyzw())
        return CylinderShape(rgba, ghost, offset, offset, radius, length)


@dataclass
class Bodies:
    """Body container"""
    bodies: List[Body]
    relative_poses: List[Body]
    pose: SE3 = field(default_factory=lambda : SE3.identity())

    @classmethod
    def from_bodies(cls, base_body:Body, other_bodies:List[Body]):
        bodies = [base_body, *other_bodies]
        rel_poses = [body.get_pose() for body in bodies]
        ref_pose = rel_poses[0]
        rel_poses = [ref_pose.inverse()@pose for pose in rel_poses]
        return cls(bodies, rel_poses, ref_pose)
    
    def get_pose(self):
        return self.pose
    
    def set_pose(self, pose:SE3):
        self.pose = pose
        poses = [self.pose@pose for pose in self.relative_poses]
        for pose, body in zip(poses, self.bodies):
            body.set_pose(pose)

@dataclass
class Frame(Bodies):
    @classmethod
    def from_pose(cls, world, name, pose=SE3.identity(), radius=0.004, length=0.04):
        viz_offsets = [
            SE3.from_xyz_xyzw([length/2,0,0],[0, 0.7071, 0, 0.7071]),
            SE3.from_xyz_xyzw([0,length/2,0],[-0.7071, 0, 0, 0.7071]),
            SE3.from_xyz_xyzw([0,0,length/2],[0, 0, 0,1]),
        ]
        cyl_shapes = [
            Cylinder.get_shape(
                radius, length, ghost=True, 
                rgba=tuple([*rgb,1.]),
                offset=offset)
            for rgb, offset in zip(np.eye(3), viz_offsets)
        ]
        axes = [Cylinder(f"{name}_{i}", axis_shape, 0., world)
                for i, axis_shape in enumerate(cyl_shapes)]
        return cls.from_bodies(axes[0], axes[1:])


#TODO: Frame
# @dataclass(frozen=True)
# class FrameShape:
#     radius: float
#     length: float

#     def get_viz_query(self, color):
#         return dict(
#             shapeType=p.GEOM_CYLINDER,
#             radius=self.radius,
#             length=self.length,
#             visualFramePosition=[0,0,self.length/2],
#             rgbaColor=[*color, 1])

# class Frame(Body):
#     axis_orns: ClassVar[np.ndarray] = np.array([
#         [0, 0.7071, 0, 0.7071],
#         [-0.7071, 0, 0, 0.7071],
#         [0,0,0,1]])
    
#     def __init__(self, world:BulletWorld, length=0.05, radius=0.005):
#         self.shape = FrameShape(radius, length) # TODO: change with 3 cylinder shapes?
#         viz_ids, col_ids = world.get_shape_id(self.shape)
#         uids = []
#         for viz_id, col_id in zip(viz_ids, col_ids):
#             uids.append(Body.create(world, viz_id, col_id))
#         super().__init__(world, uids)
#         self.pose = SE3.identity()
#         self.set_pose(self.pose)
    
#     def __del__(self):
#         for uid in self.uid:
#             self.client.removeBody(uid)
    
#     def set_pose(self, pose:SE3):
#         self.pose = pose
#         for uid, axis_orn in zip(self.uid, self.axis_orns):
#             orn = pose.rot @ SO3(axis_orn)
#             self.client.resetBasePositionAndOrientation(
#                 bodyUniqueId=uid, posObj=pose.trans, ornObj=orn.as_quat())    


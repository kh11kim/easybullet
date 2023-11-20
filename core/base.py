from __future__ import annotations

import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from .pose import SE3, SO3
from attrs import define, field
import trimesh
from typing import *
import abc
import time
from .struct import *


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
        col_id = -1 if shape.ghost \
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
    
    def get_distance_info(
        self, 
        body1:Body, 
        body2: Body, 
        link1:int=None, 
        link2:int=None,
        tol:float=0.,
    ):
        if (link1 is not None) & (link2 is not None):
            results = self.getClosestPoints(
                bodyA=body1.uid, bodyB=body2.uid, 
                linkIndexA=link1, linkIndexB=link2,
                distance=tol)
        elif (link1 is None) & (link2 is None):
            results = self.getClosestPoints(
                bodyA=body1.uid, bodyB=body2.uid, 
                distance=tol)
        return [DistanceInfo(*info) for info in results]
    
    def is_body_collision(self, body1:Body, body2:Body):
        distance_info = self.get_distance_info(body1, body2)
        return any(distance_info)


@define
class Body(abc.ABC):
    name: str
    client: BulletWorld
    uid: int = field(init=False)

    def __attrs_post_init__(self):
        assert self.name not in self.client.body_dict, "Body name already exists!"
        self.client.register_body(self)

    def set_pose(self, pose: SE3):
        self.client.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat())
    
    def get_pose(self):
        pos, orn = self.client.getBasePositionAndOrientation(self.uid)
        return SE3(SO3(orn), pos)

@define
class Geometry(Body):
    shape: Shape
    mass: float = 0.1
    viz_id: int = field(init=False)
    col_id: int = field(init=False)
    
    def __attrs_post_init__(self):
        # check if the shape is already existing
        self.viz_id, self.col_id = self.client.get_shape_id(self.shape) 
        self.uid = self.client.createMultiBody(
            baseVisualShapeIndex=self.viz_id,
            baseCollisionShapeIndex=self.col_id,
            baseMass=self.mass)
        super().__attrs_post_init__()
    
    @abc.abstractclassmethod
    def get_shape(cls):
        pass
        
@define
class Mesh(Geometry):
    viz_mesh: trimesh.Trimesh = field(init=False)

    def __attrs_post_init__(self):
        self.viz_mesh = trimesh.load(self.shape.viz_mesh_path)
        if self.shape.centering:
            self.viz_mesh.apply_translation(
                self.shape.viz_offset_xyz_xyzw[:3])
        super().__attrs_post_init__()
        
    @classmethod
    def get_shape(
        cls, viz_mesh_path:str, col_mesh_path=None, 
        rgba=None, scale=1., centering=False,
        viz_offset=None, col_offset=None
    ):
        ghost = True if col_mesh_path is None else False
        if centering:
            mesh = trimesh.load(viz_mesh_path)
            offset = mesh.bounding_box.primitive.center
            viz_offset = SE3(trans=-offset)
            col_offset = SE3(trans=-offset)
        viz_offset = viz_offset if viz_offset is not None else SE3.identity()
        col_offset = col_offset if col_offset is not None else SE3.identity()
        rgba = None if rgba is None else tuple(rgba)
        
        shape = MeshShape(
            viz_mesh_path=viz_mesh_path,
            col_mesh_path=col_mesh_path,
            scale=scale,
            centering=centering,
            viz_offset_xyz_xyzw=tuple(viz_offset.as_xyz_xyzw()),
            col_offset_xyz_xyzw=tuple(col_offset.as_xyz_xyzw()),
            rgba=rgba,
            ghost=ghost,
        )
        return shape

@define
class Cylinder(Geometry):

    @classmethod
    def get_shape(cls, radius, length, rgba=(1,1,1,1), ghost=False, offset: SE3 = None):
        offset = SE3.identity() if offset is None else offset
        offset = tuple(offset.as_xyz_xyzw())
        return CylinderShape(rgba, ghost, offset, offset, radius, length)

@define
class Box(Geometry):

    @classmethod
    def get_shape(cls, half_extents, rgba=(1,1,1,1), ghost=False, offset: SE3 = None):
        offset = SE3.identity() if offset is None else offset
        offset = tuple(offset.as_xyz_xyzw())
        return BoxShape(rgba, ghost, offset, offset, tuple(half_extents))


@define
class URDF(Body):
    path: str
    pose: SE3 = field(factory=lambda : SE3.identity())

    def __attrs_post_init__(self):
        self.uid = self.client.loadURDF(
            fileName=str(self.path),
            basePosition=self.pose.trans,
            baseOrientation=self.pose.rot.as_quat(),
            useFixedBase=True,
            # globalScaling=scale,
        )
        super().__attrs_post_init__()
        

@define
class Bodies:
    """Body container"""
    bodies: List[Body]
    relative_poses: List[Body]
    pose: SE3 = field(factory=lambda : SE3.identity())

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



@define
class Frame(Bodies):
    @classmethod
    def from_pose(cls, world, name, pose=SE3.identity(), radius=0.004, length=0.04):
        viz_offsets = [
            SE3.from_xyz_xyzw([length/2,0,0, 0, 0.7071, 0, 0.7071]),
            SE3.from_xyz_xyzw([0,length/2,0,-0.7071, 0, 0, 0.7071]),
            SE3.from_xyz_xyzw([0,0,length/2,0, 0, 0,1]),
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


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
    
    def get_contact_info(
        self, body:Body, excludes:List[Body], tol:float):
        """more expensive than distance_info due to contact force calc."""
        self.step(no_dynamics=True)
        results = self.getContactPoints(body.uid)
        results = [ContactInfo(*c) for c in results]

    def is_body_collision(self, body1:Body, body2:Body):
        distance_info = self.get_distance_info(body1, body2)
        return any(distance_info)

    def is_link_collision(self, body1:Body, body2:Body, link_idx1, link_idx2):
        distance_info = self.get_distance_info(body1, body2, link_idx1, link_idx2)
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
    
    def is_collision_with(self, other_body:Body):
        return self.client.is_body_collision(self, other_body)

    def get_dynamics_info(self, link_idx=-1):
        return DynamicsInfo(*self.client.getDynamicsInfo(self.uid, link_idx))
    
    def set_dynamics_info(self, input_dict, link_idx=-1):
        """input_dict should contain key and value of the changeDynamics()"""
        self.client.changeDynamics(
            bodyUniqueId=self.uid,
            linkIndex=link_idx,
            **input_dict)

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


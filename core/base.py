import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from .pose import SE3, SO3
from dataclasses import dataclass, field
import trimesh
from typing import *


Array = np.ndarray
ArrayLike = Union[np.ndarray,List,Tuple]


@dataclass(frozen=True)
class Shape:
    def get_query(self):
        raise NotImplementedError
    
@dataclass(frozen=True)
class FrameShape:
    radius: float
    length: float

    def get_viz_query(self, color):
        return dict(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.length,
            visualFramePosition=[0,0,self.length/2],
            rgbaColor=[*color, 1])

@dataclass(frozen=True)
class MeshShape:
    viz_mesh_path: str
    col_mesh_path: str
    scale: float
    viz_offset_xyzwxyz: Tuple[float]
    col_offset_xyzwxyz: Tuple[float]
    rgba: Union[Tuple[float], None] = None # [0, 1]

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
        self.shapes = {}
    
    def step(self, no_dynamics=False):
        if no_dynamics:
            self.performCollisionDetection()
        else:
            self.stepSimulation()

    def get_shape_id(self, shape:Shape):
        if shape in self.shapes:
            return self.shapes[shape]
        
        elif isinstance(shape, FrameShape):
            viz_id = []
            for color in np.eye(3):
                viz_id.append(self.createVisualShape(
                    **shape.get_viz_query(color)))
            col_id = (-1, -1, -1) #no collision shapes
        elif isinstance(shape, MeshShape):
            viz_id = self.createVisualShape(**shape.get_viz_query())
            col_id = -1 if shape.col_mesh_path is None \
                else self.createCollisionShape(**shape.get_col_query())
        
        self.shapes[shape] = (viz_id, col_id)
        return self.shapes[shape]
            
    
class Body:
    def __init__(self, world: BulletWorld, uid):
        self.client = world
        self.uid = uid

    def __del__(self):
        self.client.removeBody(self.uid)

    def set_pose(self, pose: SE3):
        self.client.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat())

    @classmethod
    def create(self, client, viz_shape_id:int, col_shape_id:int, mass=0.):
        return client.createMultiBody(
            baseVisualShapeIndex=viz_shape_id,
            baseCollisionShapeIndex=col_shape_id,
            baseMass=mass)

class Frame(Body):
    axis_orns: ClassVar[np.ndarray] = np.array([
        [0, 0.7071, 0, 0.7071],
        [-0.7071, 0, 0, 0.7071],
        [0,0,0,1]])
    
    def __init__(self, world:BulletWorld, length=0.05, radius=0.005):
        self.shape = FrameShape(radius, length) # TODO: change with 3 cylinder shapes?
        viz_ids, col_ids = world.get_shape_id(self.shape)
        uids = []
        for viz_id, col_id in zip(viz_ids, col_ids):
            uids.append(Body.create(world, viz_id, col_id))
        super().__init__(world, uids)
        self.pose = SE3.identity()
        self.set_pose(self.pose)
    
    def __del__(self):
        for uid in self.uid:
            self.client.removeBody(uid)
    
    def set_pose(self, pose:SE3):
        self.pose = pose
        for uid, axis_orn in zip(self.uid, self.axis_orns):
            orn = pose.rot @ SO3(axis_orn)
            self.client.resetBasePositionAndOrientation(
                bodyUniqueId=uid, posObj=pose.trans, ornObj=orn.as_quat())    

class Mesh(Body):
    def __init__(self, world: BulletWorld, 
                 viz_mesh_path, col_mesh_path=None, 
                 mass=0., rgba=None, scale=1., ghost=False, 
                 viz_offset=None, col_offset=None, centering=False):
        self.mesh = trimesh.load(viz_mesh_path)
        if centering:
            viz_offset = SE3(trans=-self.mesh.centroid)
            col_offset = SE3(trans=-self.mesh.centroid)
        viz_offset = viz_offset if viz_offset is not None else SE3.identity()
        col_offset = col_offset if col_offset is not None else SE3.identity()
        rgba = None if rgba is None else tuple(rgba)
        self.shape = MeshShape(
            viz_mesh_path=viz_mesh_path,
            col_mesh_path=col_mesh_path,
            scale=scale,
            viz_offset_xyzwxyz=tuple(viz_offset.as_xyzwxyz()),
            col_offset_xyzwxyz=tuple(col_offset.as_xyzwxyz()),
            rgba=rgba)
        
        viz_id, col_id = world.get_shape_id(self.shape)
        uid = Body.create(world, viz_id, col_id, mass)
        super().__init__(world, uid)

        


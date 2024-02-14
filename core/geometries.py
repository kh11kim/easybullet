import trimesh
from .base import BulletWorld, Body, Bodies
from .struct import *
from pathlib import Path

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
    
    @classmethod
    def create(cls, name:str, world:BulletWorld, shape:Shape, mass=0.1):
        return cls(name, world, shape, mass)
    
    
        
@define
class Mesh(Geometry):
    viz_mesh: trimesh.Trimesh = field(init=False)
    col_mesh: trimesh.Trimesh = field(init=False)

    def __attrs_post_init__(self):
        self.viz_mesh = trimesh.load(self.shape.viz_mesh_path)
        if self.shape.col_mesh_path:
            self.col_mesh = trimesh.load(self.shape.col_mesh_path)
            if self.shape.centering:
                self.viz_mesh.apply_translation(
                    self.shape.viz_offset_xyz_xyzw[:3])
        else:
            self.col_mesh = None
            if self.shape.centering:
                self.col_mesh.apply_translation(
                    self.shape.col_offset_xyz_xyzw[:3])
            
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
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(True)
            offset = mesh.centroid #mesh.bounding_box.primitive.center
            viz_offset = SE3(trans=-offset)
            col_offset = SE3(trans=-offset)
        viz_offset = viz_offset if viz_offset is not None else SE3.identity()
        col_offset = col_offset if col_offset is not None else SE3.identity()
        rgba = None if rgba is None else tuple(rgba)
        
        if isinstance(viz_mesh_path, Path): viz_mesh_path = viz_mesh_path.as_posix()
        if isinstance(col_mesh_path, Path): col_mesh_path = col_mesh_path.as_posix()
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
class Sphere(Geometry):

    @classmethod
    def get_shape(cls, radius, rgba=(1,1,1,1), ghost=False, offset: SE3 = None):
        offset = SE3.identity() if offset is None else offset
        offset = tuple(offset.as_xyz_xyzw())
        return SphereShape(rgba, ghost, offset, offset, radius=radius)

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
class Frame(Bodies):
    @classmethod
    def from_pose(cls, name, world, pose=SE3.identity(), radius=0.004, length=0.04):
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
        axes = [Cylinder(f"{name}_{i}", world, axis_shape, 0.)
                for i, axis_shape in enumerate(cyl_shapes)]
        return cls.from_bodies(axes[0], axes[1:])

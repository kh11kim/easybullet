from attrs import define, field
import pybullet as p
import trimesh
import numpy as np
import tqdm
import time
from typing import *

from .base import Body, Bodies
from .geometries import Box, Mesh
from .pose import SE3, SO3
from .urdfs import URDF
from ..assets import PANDA_HAND_URDF

@define(kw_only=True)
class Gripper(Bodies):
    tcp_offset: SE3
    max_width: float

    @property
    def hand(self)->URDF: return self.bodies[0]
    @property
    def swept_box(self)->Box: return self.bodies[1]
    @property
    def tool_pose(self)->SE3: 
        return self.hand.get_pose() @ self.tcp_offset

    @classmethod
    def create(cls, name, world, tcp_offset:SE3, max_width:float):
        hand = URDF(name, world, PANDA_HAND_URDF)
        hand_pose = tcp_offset.inverse()
        box_shape = Box.get_shape(
            half_extents=[0.0085, max_width/2, 0.0085],
            rgba=(0,1,0,0.5),
        )
        swept_box = Box(name+"_swept_box", world, box_shape)
        bodies = [hand, swept_box]
        rel_poses = [hand_pose, SE3.identity()]
        gripper = cls(
            bodies=bodies, 
            relative_poses=rel_poses, 
            pose=SE3.identity(),
            tcp_offset=tcp_offset,
            max_width=max_width)
        gripper.set_pose(SE3.identity())
        gripper.grip()
        return gripper

    def remove(self):
        remove_pose = SE3(trans=[0,0,10])
        self.set_pose(remove_pose)
    
    def grip(self, width=None, control=False, force=5):
        remove_pose = SE3(trans=[0,0,10])
        if width is None: width = self.max_width
        
        if not control:
            self.hand.set_joint_angle(0, width / 2)
            self.hand.set_joint_angle(1, width / 2)
        else:
            self.swept_box.set_pose(remove_pose)
            self.hand.client.setJointMotorControlArray(
                self.hand.uid, 
                jointIndices=[0, 1], 
                controlMode=p.POSITION_CONTROL, 
                forces=[force, force]
            )
    
    def is_graspable(self, obj:Body):
        is_in_swept_vol = self.hand.client.is_body_collision(self.swept_box, obj)
        is_col_gripper = self.hand.client.is_body_collision(self.hand, obj)
        return is_in_swept_vol and not is_col_gripper
    
@define
class GraspSim:
    obj: Mesh
    gripper: Gripper
    
    @property
    def mesh(self):
        return self.obj.col_mesh
    
    @staticmethod
    def apply_random_cone_rotation(vector, max_angle=np.pi/6):
        rand_rotvec = SO3.random().as_rotvec()
        random_rot = SO3.from_rotvec(rand_rotvec / (np.pi*2) * max_angle)
        return random_rot.apply(vector/np.linalg.norm(vector))
    
    def analyze_grasp(
        self, 
        num_samples=100_000, 
        min_depth=0.01,
        max_depth=0.03,
        visualize=True
    ):
        points, faces = trimesh.sample.sample_surface_even(self.mesh, num_samples)
        normals = self.mesh.face_normals[faces]
        
        succ_results = []
        print("Simulate grasp samples")
        for idx, (point, normal) in enumerate(zip(tqdm.tqdm(points), normals)):
            self.obj.set_pose(SE3.identity())
            self.gripper.grip()
            self.gripper.remove()
            point2 = self.get_antipodal_point(point, normal)
            if point2 is None: continue #fail
            w = np.linalg.norm(point - point2)
            if w >= self.gripper.max_width: continue
            grasp_vec = self.apply_random_cone_rotation(normal)
            cand = self.get_antipodal_grasp_pose(point, point2, grasp_vec)
            if cand is None: continue
            self.gripper.set_pose(cand)
            
            self.simulate_grasp(cand, visualize=visualize)
            result = self.check_grasp(self.obj)
            if result is None: continue
            grasp, width = result
            succ_results.append((idx, grasp, width))
        results = self.convert_to_approach_grasps(succ_results, min_depth, max_depth)
        return np.array(points), results
    
    def simulate_grasp(self, grasp_pose:SE3, force=5, visualize=True):
        self.gripper.set_pose(grasp_pose)
        for _ in range(100):
            self.gripper.grip(0, control=True, force=force)
            self.gripper.hand.client.step()
            if visualize: time.sleep(0.005)

    def check_grasp(self, obj:Mesh):
        #check
        if not self.gripper.hand.is_collision_with(obj): return None # if not holding
        width = self.gripper.hand.get_joint_angles().sum()
        # open and check
        self.gripper.grip()
        self.gripper.set_pose(self.gripper.tool_pose) 
        if not self.gripper.is_graspable(obj): return None
        final_grasp_pose = obj.get_pose().inverse() @ self.gripper.tool_pose
        return final_grasp_pose, width
    
    def get_antipodal_point(self, point:np.ndarray, normal:np.ndarray, offset=0.01):
        ray_origin = point + normal*offset
        ray_direction = -normal
        locations, _, _ = self.mesh.ray.intersects_location(
                    ray_origins=ray_origin[None, :],
                    ray_directions=ray_direction[None,:])
        if len(locations) < 2: return None
        distances = locations-ray_origin
        indices = np.argsort(np.linalg.norm(distances, axis=-1))
        p_first_hit = locations[indices][1]
        return p_first_hit
    
    def get_antipodal_grasp_pose(
        self, 
        point1:np.ndarray, 
        point2:np.ndarray, 
        grasp_vec:np.ndarray
    ):
        def get_grasp_orn_by_pitch(y, pitch):
            x_ = np.array([1,0,0])
            if np.linalg.norm(x_ - y) < 1e-4:
                x_ = np.array([0,0,1])
            z = np.cross(x_, y)
            x = np.cross(y, z)
            rotmat = np.vstack([x,y,z]).T
            rot = SO3.from_matrix(rotmat) @ SO3.from_euler("zyx", [0,pitch,0])
            return rot
        
        pitch_ref = np.random.uniform(-np.pi, np.pi)
        pitch_grid = np.linspace(0, np.pi*2, 10, endpoint=False)
        tcp = (point1 + point2)/2
        np.random.shuffle(pitch_grid)
        for pitch in pitch_grid:
            rot = get_grasp_orn_by_pitch(grasp_vec, pitch + pitch_ref)
            cand = SE3(rot, tcp)
            self.gripper.set_pose(cand)
            if self.gripper.is_graspable(self.obj):
                return cand
        return None

    def convert_to_approach_grasps(self, succ_results: List[Tuple], min_depth = 0.01, max_depth=0.03):
        print("Convert successful grasp samples to antipodal grasps")
        convert_result = []
        for grasp_idx, antipodal_grasp, width in tqdm.tqdm(succ_results):
            tcp = antipodal_grasp.trans
            approach_vec = antipodal_grasp.rot.as_matrix()[:,-1]

            ray_origin = tcp - approach_vec
            ray_direction = approach_vec
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=ray_origin[None, :],
                ray_directions=ray_direction[None,:])
            if len(locations) == 0: continue
            
            idx = np.linalg.norm(locations - ray_origin, axis=1).argmin()
            surface_point = locations[idx]
            depth = np.linalg.norm(surface_point - tcp)
            if depth >= max_depth or depth <= min_depth: continue
            
            approach_grasp = SE3(antipodal_grasp.rot, surface_point)
            convert_result.append(
                (grasp_idx, *approach_grasp.as_xyz_xyzw(), width, depth)
            )
        convert_result = np.array(convert_result)
        return convert_result   
from attrs import define, field
from typing import *

from .base import Body
from .struct import *

@define
class URDF(Body):
    path: str
    use_fixed_base: bool = True
    dof: int = field(init=False)
    info: List[JointInfo] = field(init=False)
    movable_joints: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        self.uid = self.client.loadURDF(
            fileName=str(self.path),
            useFixedBase=self.use_fixed_base,
        )
        self.dof = self.client.getNumJoints(self.uid)
        self.info = []
        movable_joints = []
        for i in range(self.dof):
            info = JointInfo(*self.client.getJointInfo(self.uid, i))
            self.info.append(info)
            if info.movable:
                movable_joints.append(i)
        self.movable_joints = np.array(movable_joints)
        super().__attrs_post_init__()
    
    def get_joint_states(self):
        return [JointState(*s) for s in self.client.getJointStates(self.uid, self.movable_joints)]
    
    def get_joint_angles(self):
        return np.array([s.pos for s in self.get_joint_states()])
    
    def set_joint_angle(self, i, angle):
        self.client.resetJointState(
            self.uid, jointIndex=i, targetValue=angle)
        
    def set_joint_angles(self, angles):
        assert len(angles) == len(self.movable_joints)
        for i, angle in zip(self.movable_joints, angles):
            self.set_joint_angle(i, angle)
    
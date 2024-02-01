from attrs import define, field
from typing import *

from .base import Body
from .struct import *

@define
class URDF(Body):
    path: str
    use_fixed_base: bool = True
    dof: int = field(init=False)
    joint_info: List[JointInfo] = field(init=False)
    movable_joints: np.ndarray = field(init=False)
    pos_ctrl_gain_p: List[float] = field(init=False)
    pos_ctrl_gain_d: List[float] = field(init=False)
    max_torque: List[float] = field(init=False)

    def __attrs_post_init__(self):
        self.uid = self.client.loadURDF(
            fileName=str(self.path),
            useFixedBase=self.use_fixed_base,
        )
        self.dof = self.client.getNumJoints(self.uid)
        self.joint_info = []
        movable_joints = []
        for i in range(self.dof):
            info = JointInfo(*self.client.getJointInfo(self.uid, i))
            self.joint_info.append(info)
            if info.movable:
                movable_joints.append(i)
        self.movable_joints = np.array(movable_joints)
        
        #default ctrl gains
        self.pos_ctrl_gain_p = [0.01] * len(self.movable_joints)
        self.pos_ctrl_gain_d = [1.0] * len(self.movable_joints)
        self.max_torque = [250] * len(self.movable_joints)
        super().__attrs_post_init__()
    
    def get_joint_states(self):
        return [JointState(*s) for s in self.client.getJointStates(self.uid, self.movable_joints)]
    
    def get_joint_angles(self):
        return np.array([s.pos for s in self.get_joint_states()])
    
    def set_joint_angle(self, i, angle):
        self.client.resetJointState(
            self.uid, jointIndex=i, targetValue=angle)
        
    def set_joint_angles(self, angles):
        assert len(angles) == len(self.movable_joints), f"num_angle is not matched: {len(angles)} vs {len(self.movable_joints)}"
        for i, angle in zip(self.movable_joints, angles):
            self.set_joint_angle(i, angle)
    
    def get_link_pose(self, link_idx):
        assert len(self.joint_info) > link_idx
        pos, xyzw = self.client.getLinkState(self.uid, link_idx)[:2]
        return SE3(SO3(xyzw), pos)
    
    def calc_ik(self, pose:SE3, link_idx:int):
        sol = self.client.calculateInverseKinematics(
            self.uid, link_idx, pose.trans, pose.rot.as_quat())
        return np.array(sol)
    
    def set_ctrl_target_joint_angles(self, q):
        assert len(q) == len(self.movable_joints)
        self.client.setJointMotorControlArray(
            self.uid, 
            jointIndices=self.movable_joints, 
            controlMode=p.POSITION_CONTROL, 
            targetPositions=q,
            forces=self.max_torque,
            positionGains=self.pos_ctrl_gain_p,
            velocityGains=self.pos_ctrl_gain_d,
        )
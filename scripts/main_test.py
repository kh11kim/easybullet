import sys
sys.path.append('../')
sys.path.append('./')
from core.base import *

world = BulletWorld(gui=True)
shape = Cylinder.get_shape(0.1, 0.5)

cylinder1 = Cylinder(world, shape, 0.1)
cylinder2 = Cylinder(world, shape, 0.1)
cylinder2.set_pose(SE3(rot=SO3.random()))

cylinders = Bodies.from_bodies(cylinder1, [cylinder2])
cylinders.set_pose(SE3(trans=[1,0,0]))
del cylinders

cylinder1 = Cylinder(world, shape, 0.1)
cylinder2 = Cylinder(world, shape, 0.1)
import sys; sys.path.append('../')
from ..core.base import *
import time

def test_bullet():
    world = BulletWorld(gui=True)
    elapsed = 0.
    dt = 0.01
    mesh_path = "/Users/math/ws/easybullet/ycb/002_master_chef_can/google_16k/textured.obj"
    #can = Mesh(world, mesh_path, mesh_path) #viz, col
    can = Mesh(world, mesh_path, centering=True) #viz only

    while True:
        world.step(no_dynamics=True)
        time.sleep(dt)
        
        elapsed += dt
        if elapsed > 5:
            break

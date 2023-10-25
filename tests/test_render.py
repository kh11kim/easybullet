from easybullet.base import *
from easybullet.render import *
import time

def test_render():
    world = BulletWorld(gui=True)
    elapsed = 0.
    dt = 0.01
    mesh_path = "/Users/math/ws/easybullet/ycb/002_master_chef_can/google_16k/textured.obj"
    #can = Mesh(world, mesh_path, mesh_path) #viz, col
    can = Mesh(world, mesh_path, centering=True) #viz only
    
    intr = CameraIntrinsic(640,480, 540,540, 320, 240, 0.1, 2)
    camera = Camera(world, intr)
    
    button_uid = p.addUserDebugParameter("quit",1,0,1)
    while True:
        world.step(no_dynamics=True)
        time.sleep(dt)
        quit = p.readUserDebugParameter(button_uid)
        if quit >= 2: break
        
        elapsed += dt
        if elapsed > 5:
            break

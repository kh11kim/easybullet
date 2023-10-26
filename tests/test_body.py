import sys; sys.path.append('../')
from ..core.base import *
import time

def test_body():
    world = BulletWorld(gui=True)
    elapsed, dt = 0., 0.01
    button_uid = p.addUserDebugParameter("quit",1,0,1)

    while True:
        world.step(no_dynamics=True)
        time.sleep(dt)
        quit = p.readUserDebugParameter(button_uid)
        if quit >= 2: break
        
        # test mesh
        # mesh_path = "/Users/math/ws/easybullet/ycb/002_master_chef_can/google_16k/textured.obj"
        # can = Mesh(world, 
        #     Mesh.get_shape(mesh_path, mesh_path, centering=True))

        #test cylinder
        # shape = Cylinder.get_shape(0.1, 0.5)
        # cylinder1 = Cylinder(world, shape, 0.1)

        #test bodies
        shape = Cylinder.get_shape(0.1, 0.5)
        cylinder1 = Cylinder(world, shape, 0.1)
        time.sleep(1)
        cylinder2 = Cylinder(world, shape, 0.1)
        cylinder2.set_pose(SE3(rot=SO3.random()))
        
        cylinders = Bodies.from_bodies(cylinder1, [cylinder2])
        time.sleep(1)
        cylinders.set_pose(SE3(trans=[1,0,0]))
        time.sleep(1)
        del cylinders
        time.sleep(1)
        #frame = Frame()




        elapsed += dt
        if elapsed > 10.:
            break

if __name__ == "__main__":
    test_body()
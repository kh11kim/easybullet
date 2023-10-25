from easybullet.pose import *

def test_SO3_generation():
    assert isinstance(SO3([1,0,0,0]), SO3)
    assert isinstance(SO3.from_euler("xyz", [1,0,0]), SO3)
    assert isinstance(SO3.from_wxyz([1,0,0,0]), SO3)
    assert isinstance(SO3.random(), SO3)
    assert isinstance(SO3.identity(), SO3)
    assert isinstance(SO3.from_matrix(np.eye(3)), SO3)

def test_SO3_matmul():
    rot = SO3.random()
    vectors = np.random.random((100,3))
    assert (rot @ vectors).shape == vectors.shape
    assert isinstance(rot@rot, SO3)
    assert np.allclose((rot.inverse() @ rot).xyzw, np.array([0,0,0,1]))

def test_SE3_generation():
    assert isinstance(SE3(), SE3)
    assert isinstance(SE3.from_matrix(np.eye(4)), SE3)

def test_SE3_matmul():
    pose = SE3.random()
    vectors = np.random.random((100,3))
    assert (pose @ vectors).shape == vectors.shape
    assert isinstance(pose@pose, SE3)
    assert np.allclose((pose.inverse() @ pose).as_matrix(), np.eye(4))

    

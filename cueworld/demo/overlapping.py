import pybullet as p
import pybullet_data
import numpy as np
import os

from cueworld.video import images_to_video
from cueworld.paths import ABS_PATH

def draw_aabb(aabbMin, aabbMax):
    """Draw a bounding box using pybullet cylinders so they show up in camera images."""
    pts = [
        [aabbMin[0], aabbMin[1], aabbMin[2]],
        [aabbMax[0], aabbMin[1], aabbMin[2]],
        [aabbMax[0], aabbMax[1], aabbMin[2]],
        [aabbMin[0], aabbMax[1], aabbMin[2]],
        [aabbMin[0], aabbMin[1], aabbMax[2]],
        [aabbMax[0], aabbMin[1], aabbMax[2]],
        [aabbMax[0], aabbMax[1], aabbMax[2]],
        [aabbMin[0], aabbMax[1], aabbMax[2]],
    ]
    edges = [
        (0, 1, 'x'), (1, 2, 'y'), (2, 3, 'x'), (3, 0, 'y'),
        (4, 5, 'x'), (5, 6, 'y'), (6, 7, 'x'), (7, 4, 'y'),
        (0, 4, 'z'), (1, 5, 'z'), (2, 6, 'z'), (3, 7, 'z')
    ]
    
    color = [1, 1, 1, 1] # White
    radius = 0.02
    
    for (i, j, axis) in edges:
        p1 = np.array(pts[i])
        p2 = np.array(pts[j])
        
        center = (p1 + p2) / 2.0
        length = np.linalg.norm(p2 - p1)
        
        if length == 0:
            continue
            
        if axis == 'x':
            orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
        elif axis == 'y':
            orn = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
        else: # 'z'
            orn = [0, 0, 0, 1]
            
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=color
        )
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_shape,
            basePosition=center,
            baseOrientation=orn
        )

def main():
    print("Starting overlapping test...")
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load some objects
    cube1 = p.loadURDF("cube.urdf", [0, 0, 0.5])
    cube2 = p.loadURDF("cube.urdf", [1, 1, 0.5])
    cube3 = p.loadURDF("cube.urdf", [-1, 0, 0.5])
    
    # Change colors to make them easily distinguishable
    p.changeVisualShape(cube1, -1, rgbaColor=[1.0, 0.0, 0.0, 1.0])  # Red
    p.changeVisualShape(cube2, -1, rgbaColor=[0.0, 1.0, 0.0, 1.0])  # Green
    p.changeVisualShape(cube3, -1, rgbaColor=[0.0, 0.0, 1.0, 1.0])  # Blue
    
    # Perform a few simulation steps to let things settle
    for _ in range(10):
        p.stepSimulation()
    
    # Define AABB
    aabbMin = [-3.0, -3.0, 0.0]
    aabbMax = [3.0, 3.0, 1.0]
    
    # Draw the bounding box
    draw_aabb(aabbMin, aabbMax)
    
    # Get overlapping objects
    overlapping = p.getOverlappingObjects(aabbMin, aabbMax)
    print(f"Bounding Box: Min {aabbMin}, Max {aabbMax}")
    print("Overlapping Objects IDs:", overlapping)
    print(f"Details: plane({plane_id}), cube1({cube1}), cube2({cube2}), cube3({cube3})")
    
    if overlapping:
        for obj_info in overlapping:
            # obj_info is a tuple: (objectUniqueId, linkIndex)
            print(f"  Object ID: {obj_info[0]}, Link Index: {obj_info[1]}")
    else:
        print("  No overlapping objects found.")
    
    # Camera settings
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[2.5, 2.5, 2],
        cameraTargetPosition=[0, 0, 0.5],
        cameraUpVector=[0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=100.0)
        
    frames = []
    
    # Render frames
    for i in range(50):
        p.stepSimulation()
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=256, 
            height=256,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
        
        # rgbImg is returned as a 1D array of RGBA pixels, reshape to (H, W, 4)
        img = np.reshape(rgbImg, (height, width, 4))[:, :, :3]  # Drop alpha channel
        img = img.astype(np.uint8)
        frames.append(img)
        
    p.disconnect()
    
    # Save video
    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/overlapping.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()

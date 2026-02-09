import os
import math
import numpy as np
import torch
import nvdiffrast
import nvdiffrast.torch as dr
import kaolin as kal
import imageio
from PIL import Image #
import sys
import datetime
import pytz # time zone
import traceback
# device = torch.device('cuda')
# glctx = nvdiffrast.torch.RasterizeGLContext(False, device='cuda') #


def elevation_azimuth_radius_to_xyz(elevation, azimuth,radius):
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    camera_position = np.array([x, y, z])
    return camera_position

def init_6_canical_cams(res, fovy, distance, device):
    '''
                  Y
                   ^
                   |
                   |---------> X
                  /
                Z
       '''
    cams = [
        # front
        kal.render.camera.Camera.from_args(eye=torch.tensor([0., 0., -distance]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),
        # back
        kal.render.camera.Camera.from_args(eye=torch.tensor([0., 0., distance]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([0., -distance, 0.]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 0, 1.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([0., distance, 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 0., 1.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),
        # left
        kal.render.camera.Camera.from_args(eye=torch.tensor([-distance, 0., 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([distance, 0., 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * fovy / 180,
                                           width=res, height=res, device=device),
    ]

    base_dirs = torch.tensor([
        [0, 0, -1.0],
        [0, 0, 1.0],
        [0, -1.0, 0],
        [0, 1.0, 0],
        [-1.0, 0, 0],
        [1.0, 0, 0],
    ], dtype=torch.float, device=device)
    return cams, base_dirs

def fibonacci_sphere(samples, radius):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_y = math.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_y * radius
        z = math.sin(theta) * radius_y * radius
        y = y * radius

        points.append((x, y, z))

    return np.array(points)

def calculate_up_vector(eye_position, target_position):
    gaze_direction = target_position - eye_position
    world_up = np.array([0, 1, 0])
    if np.allclose(np.cross(gaze_direction, world_up), 0):
        up_vector = np.array([0, 0, 1])
    else:
        side_vector = np.cross(gaze_direction, world_up)
        up_vector = np.cross(side_vector, gaze_direction)
        up_vector /= np.linalg.norm(up_vector)
    return up_vector

def create_cameras(num_views=6, distance =1.6,fovy=49.1,res =512,
                   distribution = 'fibonacci_sphere',
                   device = torch.device('cuda'),vis = False):
    '''
    :param num_views: number of views
    :param D: distance to the origin
    :return:
    '''
    if num_views == 6:
        cameras, base_dirs = init_6_canical_cams(res, fovy, distance, device)
        eye_positions = np.array([
            [0, 0, -distance],
            [0, 0, distance],
            [0, -distance, 0],
            [0, distance, 0],
            [-distance, 0, 0],
            [distance, 0, 0],
        ])
    else:
        eye_positions = fibonacci_sphere(num_views, distance)
        cameras = []
        base_dirs = torch.zeros((num_views,3),dtype=torch.float,device = device)
        up_dirs = torch.zeros((num_views,3),dtype=torch.float,device = device)
        fovy_angle = math.pi * fovy / 180
        for i,eye in enumerate(eye_positions):
            eye = np.array(eye)
            at = np.array([0,0,0])  # origin
            up = calculate_up_vector(eye, at)
            camera = kal.render.camera.Camera.from_args(eye=eye,
                                               at=at,
                                               up=up,
                                               fov=fovy_angle,
                                               width=res, height=res, device=device)
            cameras.append(camera)
            base_dir = eye-at
            base_dirs[i] = torch.tensor(base_dir).float().to(device)
            up_dirs[i] = torch.tensor(up).float().to(device)
    if vis:
        # eye_positions = np.array(eye_positions)
        base_dirs = base_dirs.detach().cpu().numpy()
        # vis_actors_vtk([get_pc_actor_vtk(pc_np=eye_positions,color=(1,0,0),point_size=10)],arrows=True)
        actors = [get_colorful_pc_actor_vtk(pc_np=eye_positions, point_size=12, opacity=1)]
        for i in range(num_views):
            actors.append(get_one_arrow_actor(center=eye_positions[i],vector=at-eye_positions[i]))
        vis_actors_vtk(actors, arrows=True)
    return cameras, eye_positions

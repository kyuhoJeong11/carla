import glob
import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random

IM_HEIGHT = 640
IM_WIDTH = 480

SHOW_CAM = False
front_camera = None
collision_hist = []

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '200.0')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

def process_img(image):
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("rgb Camera", i3)
    cv2.waitKey(1)

def seg_process_img(image):
    #tag 설정
    image.convert(carla.ColorConverter().CityScapesPalette)
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("seg Camera", i3)
    cv2.waitKey(1)

def depth_process_img(image):
    #gray scale로 변경
    #image.convert(carla.ColorConverter().Depth)
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("depth Camera", i3)
    cv2.waitKey(1)

def col_callback(colli):
    print("Collision detected:" + str(colli))
    print("Collision transform:" + str(colli.transform))
    print("Collision actor:" + str(colli.actor))
    print("Collision other_actor:" + str(colli.other_actor))
    print("Collision normal_impulse:" + str(colli.normal_impulse) + '\n\n')

def lane_callback(lane):
    print("Lane invasion detected:" + str(lane))
    print("Lane invasion transform:" + str(lane.transform))
    #print("Lane invasion actor:" + str(lane.actor))
    print("Lane invasion marking:" + str(lane.crossed_lane_markings) + '\n\n')

def obs_callback(obs):
    print("Obstacle detected:" + str(obs))
    print("Obstacle actor:" + str(obs.actor))
    print("Obstacle distance:" + str(obs.distance))
    print("Obstacle transform:" + str(obs.transform) + '\n\n')

def gnss_callback(gnss):
    print("GNSS frame: " + str(gnss.frame))
    print("GNSS timestamp: " + str(gnss.timestamp))
    print("GNSS transform: " + str(gnss.transform))
    print("GNSS lat: " + str(gnss.latitude))
    print("GNSS lng: " + str(gnss.longitude))
    print("GNSS alt: " + str(gnss.altitude) + '\n\n')

def imu_callback(imu):
    print("IMU frame:" + str(imu.frame))
    print("IMU timestamp:" + str(imu.timestamp))
    print("IMU transform:" + str(imu.transform))
    print("IMU accelerometer:" + str(imu.accelerometer))
    print("IMU gyroscope:" + str(imu.gyroscope))
    print("IMU compass:" + str(imu.compass) + '\n\n')

def rss_callback(rss):
    print("RSS Repsonse:\n" + str(rss) + '\n')

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
             ' performance but you will lose the articulated objects in the'
             ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
             ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    try:

        world = client.get_world()
        ego_vehicle = None
        ego_cam = None
        ego_col = None
        ego_lane = None
        ego_obs = None
        ego_radar = None
        ego_gnss = None
        ego_imu = None
        ego_seg = None
        ego_depth = None
        ego_rss = None
        ego_lidar = None

        # --------------
        # Start recording
        # --------------

        client.start_recorder('~/tutorial/recorder/recording01.log')

        # --------------
        # Add LIDAR sensor to ego vehicle.
        # --------------

        try:
            original_settings = world.get_settings()
            settings = world.get_settings()
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)

            delta = 0.05

            settings.fixed_delta_seconds = delta
            settings.synchronous_mode = True
            settings.no_rendering_mode = args.no_rendering
            world.apply_settings(settings)

            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(args.filter)[0]
            vehicle_transform = random.choice(world.get_map().get_spawn_points())
            vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
            vehicle.set_autopilot(args.no_autopilot)

            lidar_bp = generate_lidar_bp(args, world, blueprint_library, delta)

            user_offset = carla.Location(args.x, args.y, args.z)
            lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            point_list = o3d.geometry.PointCloud()
            '''
            # --------------
            # Add obs sensor to ego vehicle.
            # --------------

            obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
            obs_bp.set_attribute("only_dynamics", str(True))
            obs_location = carla.Location(0, 0, 0)
            obs_rotation = carla.Rotation(0, 0, 0)
            obs_transform = carla.Transform(obs_location, obs_rotation)
            ego_obs = world.spawn_actor(obs_bp, obs_transform, attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
            ego_obs.listen(lambda obs: obs_callback(obs))

            # --------------
            # Add Lane Invasion sensor to ego vehicle.
            # --------------

            lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            lane_location = carla.Location(0, 0, 0)
            lane_rotation = carla.Rotation(0, 0, 0)
            lane_transform = carla.Transform(lane_location, lane_rotation)
            ego_lane = world.spawn_actor(lane_bp, lane_transform, attach_to=vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
            ego_lane.listen(lambda lane: lane_callback(lane))

            # --------------
            # Add Collision sensor to ego vehicle.
            # --------------

            col_bp = world.get_blueprint_library().find('sensor.other.collision')
            col_location = carla.Location(0, 0, 0)
            col_rotation = carla.Rotation(0, 0, 0)
            col_transform = carla.Transform(col_location, col_rotation)
            ego_col = world.spawn_actor(col_bp, col_transform, attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
            ego_col.listen(lambda col: col_callback(col))

            # --------------
            # Add Depth sensor to ego vehicle.
            # --------------

            depth_bp = None
            depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
            depth_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
            depth_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
            depth_bp.set_attribute("fov", f"110")
            depth_location = carla.Location(2, 0, 1)
            depth_rotation = carla.Rotation(0, 0, 0)
            depth_transform = carla.Transform(depth_location, depth_rotation)
            ego_depth = world.spawn_actor(depth_bp, depth_transform, attach_to=vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)

            ego_depth.listen(lambda image: depth_process_img(image))

            # --------------
            # Add Semantic segmentation sensor to ego vehicle.
            # --------------

            seg_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            seg_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
            seg_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
            seg_bp.set_attribute("fov", f"110")
            seg_location = carla.Location(2, 0, 1)
            seg_rotation = carla.Rotation(0, 0, 0)
            seg_transform = carla.Transform(seg_location, seg_rotation)
            # seg_bp.set_attribute("sensor_tick",str(1.0))
            ego_seg = world.spawn_actor(seg_bp, seg_transform, attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
            ego_seg.listen(lambda image: seg_process_img(image))

            # --------------
            # Add RGB sensor to ego vehicle.
            # --------------

            cam_bp = None
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
            cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
            cam_bp.set_attribute("fov", f"110")
            cam_location = carla.Location(2, 0, 1)
            cam_rotation = carla.Rotation(0, 0, 0)
            cam_transform = carla.Transform(cam_location, cam_rotation)
            ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)

            ego_cam.listen(lambda image: process_img(image))
            '''
            if args.semantic:
                lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
            else:
                lidar.listen(lambda data: lidar_callback(data, point_list))

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='Carla Lidar',
                width=960,
                height=540,
                left=480,
                top=270)
            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1
            vis.get_render_option().show_coordinate_frame = True

            if args.show_axis:
                add_open3d_axis(vis)

            frame = 0
            dt0 = datetime.now()
            while True:
                if frame == 2:
                    vis.add_geometry(point_list)
                vis.update_geometry(point_list)

                vis.poll_events()
                vis.update_renderer()
                # # This can fix Open3D jittering issues:
                time.sleep(0.005)
                world.tick()

                process_time = datetime.now() - dt0
                sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                sys.stdout.flush()
                dt0 = datetime.now()
                frame += 1

        finally:
            world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)

            vehicle.destroy()
            lidar.destroy()
            vis.destroy_window()

        # --------------
        # Place spectator on ego spawning
        # --------------
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()
        spectator.set_transform(ego_vehicle.get_transform())
        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()

    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------

        client.stop_recorder()
        '''
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_radar is not None:
                ego_radar.stop()
                ego_radar.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()
            if ego_seg is not None:
                ego_seg.stop()
                ego_seg.destroy()
            if ego_depth is not None:
                ego_depth.stop()
                ego_depth.destroy()
            if ego_rss is not None:
                ego_rss.stop()
                ego_rss.destroy()

            ego_vehicle.destroy()
        '''

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')
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

class rgb1:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,90,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: rgb1.process_image(self, image))

    def process_image(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("rgb Camera1", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class rgb2:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,180,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: rgb2.process_image(self, image))

    def process_image(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("rgb Camera2", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class rgb3:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,270,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: rgb3.process_image(self, image))

    def process_image(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("rgb Camera3", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class rgb4:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: rgb4.process_image(self, image))

    def process_image(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("rgb Camera4", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class depth:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: depth.process_image(self, image))

    def process_image(self, image):
        # gray scale로 변경
        # image.convert(carla.ColorConverter().Depth)
        image.convert(carla.ColorConverter().LogarithmicDepth)

        #print("image : " + str(image.raw_data))
        i = np.array(image.raw_data)
        #print("i : " + str(i))
        #print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        '''
        #print("i : " + str(i3[0][0][2])) #r

        RGB = []

        for i in range(IM_HEIGHT) :
            for j in range(IM_WIDTH) :
                RGB.append(tuple((i3[i][j][2], i3[i][j][1], i3[i][j][0])))

        for i in range(len(RGB)):
            normalized = (RGB[i][0] + RGB[i][1] * 256 + RGB[i][2] * 256 * 256) / (256 * 256 * 256 - 1)
            in_meters = 1000 * normalized

            print(str(i) + "\nmeter : " + str(in_meters))

        '''
        cv2.imshow("depth Camera", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class seg:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', f"{IM_WIDTH}")
        bp.set_attribute('image_size_y', f"{IM_HEIGHT}")
        bp.set_attribute('fov', f"110")
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(2,0,1),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda image: seg.process_image(self, image))

    def process_image(self, image):
        # tag 설정
        image.convert(carla.ColorConverter().CityScapesPalette)
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("seg Camera", i3)
        cv2.waitKey(1)

    def get_sensor(self):
        return self.sensor

class collision:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda colli: collision.col_callback(self, colli))

    def col_callback(self, colli):
        print("Collision detected:" + str(colli))
        print("Collision transform:" + str(colli.transform))
        print("Collision actor:" + str(colli.actor))
        print("Collision other_actor:" + str(colli.other_actor))
        print("Collision normal_impulse:" + str(colli.normal_impulse) + '\n\n')

    def get_sensor(self):
        return self.sensor

class obstacle:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda obs: obstacle.obs_callback(self, obs))

    def obs_callback(self, obs):
        print("Obstacle detected:" + str(obs))
        print("Obstacle actor:" + str(obs.actor))
        print("Obstacle distance:" + str(obs.distance))
        print("Obstacle transform:" + str(obs.transform) + '\n\n')

    def get_sensor(self):
        return self.sensor

class lane:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        print(self._parent)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda inv: lane.lane_callback(self, inv))

    def lane_callback(self, inv):
        print("Lane invasion detected:" + str(inv))
        print("Lane invasion transform:" + str(inv.transform))
        print("Lane invasion marking:" + str(len(inv.crossed_lane_markings)) + '\n\n')
        if len(inv.crossed_lane_markings) > 2:
            print("Lane invasion actor:" + str(inv.actor))

    def get_sensor(self):
        return self.sensor

class gnss:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda data: gnss.gnss_callback(self, data))

    def gnss_callback(self, data):
        print("GNSS frame: " + str(data.frame))
        print("GNSS timestamp: " + str(data.timestamp))
        print("GNSS transform: " + str(data.transform))
        print("GNSS lat: " + str(data.latitude))
        print("GNSS lng: " + str(data.longitude))
        print("GNSS alt: " + str(data.altitude) + '\n\n')

    def get_sensor(self):
        return self.sensor

class imu:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(0,0,0),carla.Rotation(0,0,0)), attach_to=self._parent)
        self.sensor.listen(lambda data: imu.imu_callback(self, data))

    def imu_callback(self, data):
        print("IMU frame:" + str(data.frame))
        print("IMU timestamp:" + str(data.timestamp))
        print("IMU transform:" + str(data.transform))
        print("IMU accelerometer:" + str(data.accelerometer))
        print("IMU gyroscope:" + str(data.gyroscope))
        print("IMU compass:" + str(data.compass) + '\n\n')

    def get_sensor(self):
        return self.sensor

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        #default='192.168.80.55',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    try:

        world = client.get_world()

        ego_cam1 = None
        ego_cam2 = None
        ego_cam3 = None
        ego_cam4 = None
        ego_depth = None
        ego_seg = None
        ego_col = None
        ego_obs = None
        ego_lane = None
        ego_gnss = None
        ego_imu = None

        # --------------
        # Start recording
        # --------------

        print("Recording on file : %s " % client.start_recorder('test2.log', True))

        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color', ego_color)
        print('\nEgo color is set')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            print(ego_transform)
            ego_vehicle.set_autopilot(True)
            print('\nEgo is spawned')
        else:
            logging.warning('Could not found any spawn points')


        rgb_cam1 = rgb1(ego_vehicle)
        ego_cam1 = rgb_cam1.get_sensor()

        rgb_cam2 = rgb2(ego_vehicle)
        ego_cam2 = rgb_cam2.get_sensor()

        rgb_cam3 = rgb3(ego_vehicle)
        ego_cam3 = rgb_cam3.get_sensor()

        rgb_cam4 = rgb4(ego_vehicle)
        ego_cam4 = rgb_cam4.get_sensor()

        '''
        depth_cam = depth(ego_vehicle)
        ego_depth = depth_cam.get_sensor()

        seg_cam = seg(ego_vehicle)
        ego_seg = seg_cam.get_sensor()
        
        col_sen = collision(ego_vehicle)
        ego_col = col_sen.get_sensor()
        
        obs_sen = obstacle(ego_vehicle)
        ego_obs = obs_sen.get_sensor()
        
        lane_sen = lane(ego_vehicle)
        ego_lane = lane_sen.get_sensor()
        
        gnss_sen = gnss(ego_vehicle)
        ego_gnss = gnss_sen.get_sensor()

        imu_sen = imu(ego_vehicle)
        ego_imu = imu_sen.get_sensor()
        '''

        # --------------
        # Check FPS
        # --------------

        frame = 0
        dt0 = datetime.now()
        while True:
            time.sleep(0.005)
            world.tick()

            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

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
        if ego_vehicle is not None:
            if ego_cam1 is not None:
                ego_cam1.stop()
                ego_cam1.destroy()
            if ego_cam2 is not None:
                ego_cam2.stop()
                ego_cam2.destroy()
            if ego_cam3 is not None:
                ego_cam3.stop()
                ego_cam3.destroy()
            if ego_cam4 is not None:
                ego_cam4.stop()
                ego_cam4.destroy()
            if ego_depth is not None:
                ego_depth.stop()
                ego_depth.destroy()
            if ego_seg is not None:
                ego_seg.stop()
                ego_seg.destroy()
            if ego_col is not None:
                ego_col.stop()
                ego_col.destroy()
            if ego_obs is not None:
                ego_obs.stop()
                ego_obs.destroy()
            if ego_lane is not None:
                ego_lane.stop()
                ego_lane.destroy()
            if ego_gnss is not None:
                ego_gnss.stop()
                ego_gnss.destroy()
            if ego_imu is not None:
                ego_imu.stop()
                ego_imu.destroy()

            ego_vehicle.destroy()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')
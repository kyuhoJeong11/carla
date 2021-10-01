#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import threading

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
from numpy import random

actor_list1 = []
actor_list2 = []
vehicle_list1 = []
vehicle_list2 = []
right = True
right_neub = None
left_neub = None

class Worker(threading.Thread) :
    def __init__(self, world1, world2, blueprint_library1, blueprint_library2, client1, client2):
        super().__init__()
        # thread에서 사용할 world, blueprint library, client 정보 초기화.
        self.world1 = world1
        self.world2 = world2
        self.blueprint_library1 = blueprint_library1
        self.blueprint_library2 = blueprint_library2
        self.client1 = client1
        self.client2 = client2
        print("thread started")

    def run(self):
        # 전역변수를 지역 내에서 사용하기 위해 global을 통해 선언
        global right
        global actor_list1
        global actor_list2
        global vehicle_list1
        global vehicle_list2
        global right_neub
        global left_neub

        # angular_velocity, velocity를 초기화 하기 위해 사용.
        vecInit = carla.Vector3D(0.0, 0.0, 0.0)

        # test를 위해 우측 배달 로봇에 autopilot ON
        right_neub.set_autopilot(True)

        while True:
            a = input("press Enter key to change world")

            if a == "":
                # 우측통행 Carla에서.
                if right is True:
                    print("start world 2")

                    # 현재 배달 로봇의 physics 값을 가져옴.
                    ang_vel = right_neub.get_angular_velocity()
                    vel = right_neub.get_velocity()
                    control = right_neub.get_control()
                    transform = right_neub.get_transform()
                    transform.location.z += 0.2

                    # 가져온 physics 값들을 좌측 통행 배달로봇에 적용.
                    left_neub.set_target_angular_velocity(ang_vel)
                    left_neub.set_target_velocity(vel)
                    left_neub.apply_control(control)
                    left_neub.set_transform(transform)

                    control.throttle = 0.0
                    control.steer = 0.0
                    control.brake = 1.0
                    control.hand_brake = False
                    control.reverse = False
                    control.manual_gear_shift = False
                    control.gear = 0

                    # 우측 통행의 autopilot OFF 이후 우측 통행 배달 로봇이 움직이지 않도록 설정.
                    right_neub.set_autopilot(False)

                    right_neub.apply_control(control)
                    right_neub.set_target_angular_velocity(vecInit)
                    right_neub.set_target_velocity(vecInit)

                    # 이후 좌측 통행에서 autopilot을 실행해야 하지만, 현재 오류가 있어서 임시로 주석처리 해놓음.
                    #left_neub.set_autopilot(True)

                    right = False

                    '''
                    # actor list, vehicle list 저장.
                    actor_list1 = self.world1.get_actors()
                    vehicle_list1 = actor_list1.filter('vehicle.*')

                    for vehicle in vehicle_list1:
                        bp = self.blueprint_library2.filter(vehicle.type_id)[0]

                        # vehicle 정보 저장
                        #attr = vehicle.attributes
                        ang_vel = vehicle.get_angular_velocity()
                        vel = vehicle.get_velocity()
                        transform = vehicle.get_transform()
                        transform.location.z += 0.5
                        control = vehicle.get_control()
                        phys_control = vehicle.get_physics_control()

                        # 각각의 attributes도 적용을 하려고 했으나, read only로 설정되어 있는 항목들이 있음.

                        #for key, val in attr.items():
                            #bp.set_attribute(key, val)


                        #vehicle.set_autopilot(False)

                        # 기존의 actor 파괴 후 다른 좌측통행 Carla에 새 actor 생성. 이후 저장된 정보 적용
                        self.client1.apply_batch([carla.command.DestroyActor(vehicle)])

                        moved_vehicle = self.world2.try_spawn_actor(bp, transform)

                        #moved_vehicle.set_target_velocity(vel)
                        #moved_vehicle.set_target_angular_velocity(ang_vel)
                        #moved_vehicle.apply_control(control)
                        #moved_vehicle.apply_physics_control(phys_control)

                        # autopilot 실행 시 client shut down, actor 파괴 등의 문제가 있음.
                        #moved_vehicle.set_autopilot(True)

                    right = False

                    actor_list1 = None
                    vehicle_list1 = None
                    '''

                    print("stop world 1")

                # 좌측통행 Carla에서.
                else:
                    print("start world 1")

                    '''
                    # actor list, vehicle list 저장.
                    actor_list2 = self.world2.get_actors()
                    vehicle_list2 = actor_list2.filter('vehicle.*')

                    for vehicle in vehicle_list2:
                        bp = self.blueprint_library1.filter(vehicle.type_id)[0]

                        # vehicle 정보 저장.
                        #attr = vehicle.attributes
                        ang_vel = vehicle.get_angular_velocity()
                        vel = vehicle.get_velocity()
                        transform = vehicle.get_transform()
                        transform.location.z += 0.5
                        control = vehicle.get_control()
                        phys_control = vehicle.get_physics_control()

                        # 각각의 attributes도 적용을 하려고 했으나, read only로 설정되어 있는 항목들이 있음.
                        
                        
                        #for key, val in attr.items():
                            #bp.set_attribute(key, val)
                        

                        #vehicle.set_autopilot(False)

                        # 기존의 actor 파괴 후 다른 좌측통행 Carla에 새 actor 생성. 이후 저장된 정보 적용
                        self.client2.apply_batch([carla.command.DestroyActor(vehicle)])

                        moved_vehicle = self.world1.try_spawn_actor(bp, transform)

                        #moved_vehicle.set_target_velocity(vel)
                        #moved_vehicle.set_target_angular_velocity(ang_vel)
                        #moved_vehicle.apply_control(control)
                        #moved_vehicle.apply_physics_control(phys_control)

                        # autopilot 실행 시 client shut down, actor 파괴 등의 문제가 있음.
                        #moved_vehicle.set_autopilot(True)

                    right = True

                    actor_list2 = None
                    vehicle_list2 = None
                    '''

                    # 현재 배달 로봇의 physics 값을 가져옴.
                    ang_vel = left_neub.get_angular_velocity()
                    vel = left_neub.get_velocity()
                    control = left_neub.get_control()
                    transform = left_neub.get_transform()
                    transform.location.z += 0.2

                    # 가져온 physics 값들을 우측 통행 배달로봇에 적용.
                    right_neub.set_target_angular_velocity(ang_vel)
                    right_neub.set_target_velocity(vel)
                    right_neub.apply_control(control)
                    right_neub.set_transform(transform)

                    control.throttle = 0.0
                    control.steer = 0.0
                    control.brake = 1.0
                    control.hand_brake = False
                    control.reverse = False
                    control.manual_gear_shift = False
                    control.gear = 0

                    # 좌측 통행의 autopilot OFF 이후 좌측 통행 배달 로봇이 움직이지 않도록 설정.
                    left_neub.set_autopilot(False)

                    left_neub.apply_control(control)
                    left_neub.set_target_angular_velocity(vecInit)
                    left_neub.set_target_velocity(vecInit)

                    # 이후 우측 통행에서 autopilot을 실행해야 하지만, 현재 오류가 있어서 임시로 주석처리 해놓음.
                    #right_neub.set_autopilot(True)

                    right = True

                    print("stop world 2")

def main():
    # args를 관리함. client, port의 default값을 변경 가능.
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-c1', '--client1',
        metavar='C',
        default='192.168.80.34',
        help='IP of Client1 (default: 192.168.80.34)')
    argparser.add_argument(
        '-p1', '--port1',
        metavar='P',
        default=2000,
        type=int,
        help='Port of Client1 (default: 2000)')
    argparser.add_argument(
        '-c2', '--client2',
        metavar='C',
        default='192.168.80.55',
        help='IP of Client2 (default: 192.168.80.55)')
    argparser.add_argument(
        '-p2', '--port2',
        metavar='P',
        default=2000,
        type=int,
        help='Port of Client2 (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    args = argparser.parse_args()

    global right_neub
    global left_neub

    # 각각 우측/좌측 통행 client에 접속.
    client1 = carla.Client(args.client1, args.port1)
    client1.set_timeout(30.0)
    client2 = carla.Client(args.client2, args.port2)
    client2.set_timeout(30.0)

    try:
        # 좌측/우측 통행 Carla의 world에 접근. 이후 각 world에서 blueprint library 호출.
        world1 = client1.get_world()
        blueprint_library1 = world1.get_blueprint_library()
        world2 = client2.get_world()
        blueprint_library2 = world2.get_blueprint_library()

        # 우측 통행 map에 배달 로봇 생성.
        bp = blueprint_library1.filter('neubility')[0]
        spawn_point = world1.get_map().get_spawn_points()[0]

        right_neub = world1.try_spawn_actor(bp, spawn_point)

        # 좌측 통행 map에 배달 로봇 생성.
        bp = blueprint_library2.filter('neubility')[0]
        spawn_point = world2.get_map().get_spawn_points()[0]

        left_neub = world2.try_spawn_actor(bp, spawn_point)

        print("bridge started")

        # 프로그램 실행 중 키 입력을 위해 thread 사용.
        t = Worker(world1, world2, blueprint_library1, blueprint_library2, client1, client2)
        t.daemon = True
        t.start()

        while True:
            time.sleep(0.005)
            world1.tick()
            world2.tick()

    finally:
        time.sleep(0.5)
        right_neub.destroy()
        left_neub.destroy()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

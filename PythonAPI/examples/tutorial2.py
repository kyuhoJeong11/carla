import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from datetime import datetime

'''
Carla 패키지가 사용하는 egg파일 탐색
'''
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []

# 서버 접속 및 차량
try:
    client = carla.Client("localhost", 2000)  # 서버 접속
    client.set_timeout(20.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()  # blueprint access

    bp = blueprint_library.filter("neubility")[0]  # spawn할 vehicle의 bp 가져오기
    spawn_point = world.get_map().get_spawn_points()[2]

    vehicle = world.spawn_actor(bp, spawn_point)  # 차 spawn

    # fl = front left / fr = front right / bl = back left / br = back right
    vehicle.apply_control(carla.VehicleControl(fl=10.0, fr=10.0, bl=10.0, br=10.0))  # 차 제어.
    actor_list.append(vehicle)  # actor list에 차량 추가.


    bp = blueprint_library.filter("neubility")[0]  # spawn할 vehicle의 bp 가져오기

    spawn_point = world.get_map().get_spawn_points()[3]

    vehicle2 = world.spawn_actor(bp, spawn_point)  # 차 spawn

    vehicle2.apply_control(carla.VehicleControl(fl=20.0, fr=20.0, bl=20.0, br=20.0))  # 차 제어.
    actor_list.append(vehicle2)  # actor list에 차량 추가.

    # vehicle의 control 정보를 가져와서 필요할 때 마다 수정하여 적용 가능.
    '''
    control = vehicle.get_control()

    control.fl = 10.0
    control.fr = 200.0
    control.bl = 10.0
    control.br = 200.0

    vehicle.apply_control(control)
    '''

    while True:
        time.sleep(0.005)
        world.tick()

# 서버 종료 시 액터 파괴. 파괴하지 않는다면 서버에 액터가 계속 존재함.
finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
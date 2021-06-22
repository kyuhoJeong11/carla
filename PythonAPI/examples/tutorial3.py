'''

.step(action)

'''

'''
def reset(self):


def step(self, action):
    return obs, reward, done, extra_info

'''

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10

'''
환경 class 세팅
'''
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW # 미리보기 여부
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    actor_list = []
    collision_hist = [] # collision 목록

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        # client가 켜져 있다면, world 검색 가능.
        self.world = self.client.get_world()
        # world에는 우리가 시뮬레이션에 액터를 새로 추가할 때 사용할 수 있는 bp 목록이 있다.
        blueprint_library = self.world.get_blueprint_library()
        # 차량 모델 지정
        self.model_3 = blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # 랜덤한 위치에 차량 생성 후 actor list에 추가
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # rgb Camera 센서의 bp 가져오기
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')

        # rgb Camera 센서로 입력 받은 이미지의 크기 조절
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        #sensor의 위치 조정
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # 센서의 생성 및 리스트 추가.
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)

        # 센서로 입력 받은 데이터를 활용하기 위해 lambda 함수 사용
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        '''
        차량 생성 시 차가 지면에 부딪히면 충돌이 발생. 
        또는 센서들이 초기화되고 값을 반환하는 데 시간이 걸릴 수 있음. 
        따라서 4초 정도의 대기시간을 사용.
        '''

        # collision 센서의 bp 가져오기
        colsensor = self.bluprint_library.find("sensor.other.collision")

        # 센서의 생성 및 리스트 추가
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)

        # 센서로 입력 받은 데이터를 활용하기 위해 lambda 함수 사용
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        '''
        에피소드의 실제 확인 시간 기록.
        브레이크와 스로틀이 사용되지 않는지 확인 후
        첫 번째 관찰 결과 반환.        
        '''
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera

    # collision data 처리
    def collision_data(self, event):
        self.collision_hist.append(event)

    # image data 처리
    def process_img(self, image):
        i = np.array(image.raw_data)
        print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    # action, reward, done, any_extra_info 관리
    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        elif kmh < 50:
            done = False
            reward = -1

        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None
import gymnasium as gym
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.wrappers import LoggerWrapper

from pyenergyplus.api import EnergyPlusAPI

# EnergyPlusAPI 사용
api = EnergyPlusAPI()

# 환경 등록
gym.envs.registration.register(
    id='CustomOffice-v0',
    entry_point=EplusEnv,
    kwargs={
        'building_file': '/Users/seoah/PycharmProjects/global-frontier-24/sinergym_env/lib/python3.12/site-packages/sinergym/data/buildings/ASHRAE901_OfficeLarge_STD2019_Denver.epJSON',
        'weather_files': ['KOR_Inchon.471120_IWEC.epw'],
        'reward_kwargs': {
            'temperature_variables': ['ZoneAirTemperature'],  # 상태에서 찾은 온도 변수명
            'energy_variables': ['Facility Total HVAC Electricity Demand Rate'],  # 예시 값
            'range_comfort_winter': [20.0, 24.0],  # 겨울철 적정 온도 범위
            'range_comfort_summer': [24.0, 28.0]  # 여름철 적정 온도 범위
        }
    }
)

# 환경 생성
env = gym.make('CustomOffice-v0')

# 상태를 확인
state = env.reset()  # 초기 상태 가져오기
print("Initial state:", state)  # 초기 상태 출력하여 온도 변수 확인

# 상태에서 사용 가능한 키 확인
print("Keys in state:", state[1].keys())  # state[1]에는 상태 정보가 포함되어 있음

# 환경 종료
env.close()
import logging
import numpy as np
from datetime import datetime
import gymnasium as gym
from typing import List, Sequence, Any

from sinergym.envs import EplusEnv
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
from sinergym.utils.rewards import LinearReward

import wandb


wandb.init(
    project="global-frontier",
    entity="seoah-ewha-womans-university",
    name="Fixed-Control-HVAC",
    group="Fixed_Control",
    tags=["Fixed", "Control", "HVAC", "5zone"],
)

building_file = "/Users/seoah/PycharmProjects/global-frontier-24/ASHRAE901_OfficeLarge_STD2019_Denver.epJSON"
weather_file = "/Users/seoah/PycharmProjects/global-frontier-24/sinergym_env/lib/python3.12/site-packages/sinergym/data/weather/KOR_Inchon.471120_IWEC.epw"
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = f'DDPG-LargeOffice-{experiment_date}'

env = EplusEnv(
building_file=building_file,
weather_files=[weather_file],
action_space=gym.spaces.Box(
    low=np.array([18.0, 22.0]),
    high=np.array([24.0, 30.0]),
    dtype=np.float32
),
variables={
    "Outdoor Air Temperature": ("Site Outdoor Air Drybulb Temperature", "ENVIRONMENT"),
    "Zone Mean Air Temperature": ("Zone Mean Air Temperature", "CORE_BOTTOM"),
    "Facility Total HVAC Electricity Demand Rate": ("Facility Total HVAC Electricity Demand Rate", "WHOLE BUILDING")
},
meters={},
actuators={
    "Heating Setpoint": ("Schedule:Compact", "Schedule Value", "HTGSETP_SCH_NO_OPTIMUM"),
    "Cooling Setpoint": ("Schedule:Compact", "Schedule Value", "CLGSETP_SCH_NO_SETBACK")
},
reward=LinearReward,
reward_kwargs={
"temperature_variables": ["Zone Mean Air Temperature"],
"energy_variables": ["Facility Total HVAC Electricity Demand Rate"],
"range_comfort_winter": (20.0, 23.5),
"range_comfort_summer": (23.0, 26.0)
}
)


observation_variables = env.get_wrapper_attr(
        'variables')

action_variables = env.get_wrapper_attr(
        'actuators')

setpoints_summer = np.array((23.0, 26.0), dtype=np.float32)
setpoints_winter = np.array((20.0, 23.5), dtype=np.float32)


# env = NormalizeObservation(env)
# env = NormalizeAction(env)

class CustomZone(object):
    building_file = "ASHRAE901_OfficeLarge_STD2019_Denver.epJSON"
    weather_file = "KOR_Inchon.471120_IWEC.epw"
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = f'DDPG-LargeOffice-{experiment_date}'

    env = env
    
    observation_variables = env.get_wrapper_attr(
            'variables')
    
    action_variables = env.get_wrapper_attr(
            'actuators')

    setpoints_summer = np.array((23.0, 26.0), dtype=np.float32)
    setpoints_winter = np.array((20.0, 23.5), dtype=np.float32)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))
        year = 2024
        month = obs_dict.get('month', 1)  # 기본값: 1월
        day = obs_dict.get('day_of_month', 1)  # 기본값: 1일

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:  # pragma: no cover
            season_range = self.setpoints_summer
        else:  # pragma: no cover
            season_range = self.setpoints_winter

        return season_range

class MyRuleBasedController(CustomZone):

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on outdoor air drybulb temperature and daytime.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        
        obs_dict = dict(zip(self.env.get_wrapper_attr(
            'variables'), observation))
        
        act_dict = dict(zip(self.env.get_wrapper_attr(
            'actuators'), observation))
        
        print(obs_dict)

        out_temp = obs_dict['Outdoor Air Temperature']
        year = 2024  # 연도를 고정하거나 필요시 obs_dict에서 동적으로 가져오도록 수정
        month = obs_dict.get('month', 1)  # 기본값: 1월
        day = obs_dict.get('day_of_month', 1)  # 기본값: 1일

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            season_comfort_range = self.setpoints_summer
        else:
            season_comfort_range = self.setpoints_summer
        season_comfort_range = self.setpoints_winter
        # Update setpoints
        in_temp = obs_dict['Zone Mean Air Temperature']

        current_heat_setpoint = act_dict[
            'Heating Setpoint']
        current_cool_setpoint = act_dict[
            'Cooling Setpoint']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if in_temp < season_comfort_range[0]:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif in_temp > season_comfort_range[1]:
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        # Clip setpoints to the action space
        if new_heat_setpoint > self.env.get_wrapper_attr('action_space').high[0]:
            new_heat_setpoint = self.env.get_wrapper_attr(
                'action_space').high[0]
        if new_heat_setpoint < self.env.get_wrapper_attr('action_space').low[0]:
            new_heat_setpoint = self.env.get_wrapper_attr(
                'action_space').low[0]
        if new_cool_setpoint > self.env.get_wrapper_attr('action_space').high[1]:
            new_cool_setpoint = self.env.get_wrapper_attr(
                'action_space').high[1]
        if new_cool_setpoint < self.env.get_wrapper_attr('action_space').low[1]:
            new_cool_setpoint = self.env.get_wrapper_attr(
                'action_space').low[1]

        action = (new_heat_setpoint, new_cool_setpoint)
        if current_dt.weekday() > 5:
            action = (
                2 * (np.array([18.33, 23.33]) - self.env.get_wrapper_attr('action_space').low) /
                (self.env.get_wrapper_attr('action_space').high - self.env.get_wrapper_attr('action_space').low) - 1
            )

        return action

agent = MyRuleBasedController()

# 에피소드 반복 설정
num_episodes = 15  # 에피소드 횟수 설정

for episode in range(1, num_episodes + 1):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    step_count = 0
    cum_hvac = 0  # 누적 HVAC 소비량 저장 리스트

    print(f"🚀 Starting Episode {episode}...")

    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        # HVAC 소비량 추출 및 누적 계산
        obs_dict = dict(zip(env.get_wrapper_attr('variables'), obs))
        hvac_demand = obs_dict['Facility Total HVAC Electricity Demand Rate']
        cum_hvac += hvac_demand
        step_count += 1

        # ✅ Step별 WandB 로깅
        wandb.log({
            "Step": step_count,
            "HVAC Demand": hvac_demand
        }, step=step_count)

        print(f"🔍 Step {step_count}: HVAC Demand = {hvac_demand}")

    # ✅ 에피소드 평균 계산 및 WandB 로깅
    mean_hvac_demand = cum_hvac / step_count if step_count > 0 else 0
    wandb.log({
        "Episode": episode,
        "Avg Facility HVAC Demand": mean_hvac_demand
    }, step=episode)

    print(f"✅ Episode {episode} Completed: Avg HVAC Demand = {mean_hvac_demand:.2f}")

env.close()
wandb.finish()
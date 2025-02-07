import logging
import gymnasium as gym
import numpy as np
import wandb
from sinergym.envs.eplus_env import EplusEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from sinergym.utils.wrappers import *
from sinergym.utils.logger import *
from datetime import datetime
import sinergym
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *


# Logger 설정
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(name='MAIN', level=logging.INFO)

# Environment ID
environment = 'Eplus-5zone-mixed-continuous-stochastic-v1'

# Training episodes 
episodes = 30

# Name of the experiment
# 환경 설정
reward_kwargs = {
    "temperature_variables": ["Zone Mean Air Temperature"],
    "energy_variables": ["Facility Total HVAC Electricity Demand Rate"],
    "range_comfort_winter": (20.0, 23.5),
    "range_comfort_summer": (23.0, 26.0)
}

building_file = "ASHRAE901_OfficeLarge_STD2019_Denver.epJSON"
weather_file = "KOR_Inchon.471120_IWEC.epw"
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = f'DDPG-LargeOffice-{experiment_date}'

# 학습 환경 생성
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
    reward_kwargs=reward_kwargs
)

env = NormalizeObservation(env)
env = NormalizeAction(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

env = WandBLogger(env,
                 entity='seoah-ewha-womans-university',
                 project_name='global-frontier',
                 run_name=experiment_name,
                 group='Train_example',
                 tags=['DRL', 'PPO', '5zone', 'continuous', 'stochastic', 'v1'],
                 save_code = True,
                 dump_frequency = 1000,
                 artifact_save = False)

eval_env = EplusEnv(
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
    reward_kwargs=reward_kwargs
)

eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)

# PPO 알고리즘 학습
model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=43,
    learning_rate=0.003
)

class WandBLoggingCallback(BaseCallback):
    """
    Custom callback to log both step-wise and episode-wise HVAC power consumption.
    """

    def __init__(self, verbose=0):
        super(WandBLoggingCallback, self).__init__(verbose)
        self.episode_hvac_demand = []  # 에피소드별 HVAC 소비량 저장 리스트
        self.episode_count = 0  # 에피소드 카운터

    def _on_step(self) -> bool:
        """Step 단위로 데이터를 모으지만, 기록은 하지 않음."""
        logger = self.training_env.get_attr('logger')[0]  # LoggerWrapper 객체 가져오기

        if hasattr(logger, "get_log_data"):
            log_data = logger.get_log_data()

            # HVAC 전력 소비량 저장 (step 단위 & 에피소드 단위)
            if "Facility Total HVAC Electricity Demand Rate" in log_data:
                hvac_value = log_data["Facility Total HVAC Electricity Demand Rate"]
                self.episode_hvac_demand.append(hvac_value)  # 에피소드별 리스트에 저장

        return True  # 학습 계속 진행

    def _on_rollout_end(self) -> None:
        if len(self.episode_hvac_demand) > 0:
            avg_hvac_demand = np.mean(self.episode_hvac_demand)  # 에피소드별 평균 계산

            print(f"[Episode {self.episode_count}] Avg HVAC Demand: {avg_hvac_demand}")

            wandb.log({
                "Episode": self.episode_count,  # 명확하게 Episode 값을 WandB에 로깅
                "Avg Facility HVAC Demand": avg_hvac_demand
            }, step=self.episode_count)  # Step을 에피소드 번호로 설정

            # 데이터 초기화 (다음 에피소드 데이터 저장)
            self.episode_hvac_demand = []
            self.episode_count += 1  # 에피소드 증가




# wandb 커스텀 콜백
if is_wrapped(env, WandBLogger):
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    # experiment_params.update(conf)
    env.get_wrapper_attr('wandb_run').config.update(experiment_params)

    callbacks = []

# Set up Evaluation logging and saving best model
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    n_eval_episodes=1,
    eval_freq_episodes=2,
    deterministic=True)

callbacks.append(eval_callback)

hvac_logging_callback = WandBLoggingCallback()
callbacks.append(hvac_logging_callback)

callback = CallbackList(callbacks)

# wandb logger and setting in SB3
if is_wrapped(env, WandBLogger):
    logger = SB3Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(
                sys.stdout,
                max_length=120),
            WandBOutputFormat()])
    model.set_logger(logger)

# 학습
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)   

model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=100)   

model.save(env.get_wrapper_attr('workspace_path') + '/model')  

env.close()

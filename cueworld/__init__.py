# from gymnasium.envs.registration import register
# from cueworld.base import BaseEnv
# from cueworld.joint_attention_wrappers import *
# from cueworld.object_wrappers import *
# from cueworld.dyadic_wrappers import *
# from gymnasium import Env

# import warnings
# warnings.filterwarnings("ignore", module="gymnasium.utils.passive_env_checker")
# warnings.filterwarnings("ignore", module="gymnasium.spaces.box")

# class MultiCueWorld(Env):
#     def __init__(self, env_types, *args, **kwargs):
#         context = {} #下位ラッパーで上位ラッパーの内容を使いたいときに使用
#         env = BaseEnv(*args, **kwargs)
#         if "rja" in env_types:
#             env = RJAWrapper(env, *args, **kwargs)
#         elif "navigation" in env_types:
#             env = NavigationWrapper(env, *args, **kwargs)
#         elif "dyadic" in env_types:
#             env = DyadicWrapper(env, *args, **kwargs)
#         elif "ija" in env_types:
#             env = InitiatingJointAttentionRewardWrapper(env, *args, **kwargs)
#             env = IJAParentWrapper(env, *args, **kwargs)
#         self.env = env
        
#         self.name = "MultiCueWorld"
    
#     def __getattr__(self, name):
#         return getattr(self.env, name)

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return obs, info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         return obs, reward, terminated, truncated, info

# #環境登録
# register(
#     id="rja",
#     entry_point="cueworld:MultiCueWorld",
#     kwargs={"env_types": ["rja"]}
# )
# register(
#     id="ija",
#     entry_point="cueworld:MultiCueWorld",
#     kwargs={"env_types": ["ija"]}
# )
# register(
#     id="navigation",
#     entry_point="cueworld:MultiCueWorld",
#     kwargs={"env_types": ["navigation"]}
# )
# register(
#     id="dyadic",
#     entry_point="cueworld:MultiCueWorld",
#     kwargs={"env_types": ["dyadic"]}
# )
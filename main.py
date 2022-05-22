from trainers.dqn_epsilon_keras import train
from models.deepmind_atari import create_model
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)

model = create_model(env.observation_space.shape, env.action_space.n)
model_target = create_model(env.observation_space.shape, env.action_space.n)

train(model, model_target, env)

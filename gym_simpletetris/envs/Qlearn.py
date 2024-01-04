import numpy as np
import gym
import tensorflow as tf
import gym_simpletetris
import pygame


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from tensorflow import keras
from keras import __version__
tf.keras.__version__ = __version__

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy

# Create the Tetris environment
env = gym.make("SimpleTetris-v0",   )
obs = env.reset()

pygame.init()

states = env.observation_space.shape[0]
actions = env.action_space.n

print(states)
print(actions)


# Define the neural network model for the DQN
model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions, activation='linear'))


# Create the DQN agent
Agent = DQNAgent(model=model, 
                 nb_actions=actions, 
                 memory = SequentialMemory(limit=5000, window_length=1),
                 nb_steps_warmup=100,
                 target_model_update=1e-2, 
                 policy=EpsGreedyQPolicy(eps=0.1)
            )

# Compile the model using TensorFlow's Adam optimizer
Agent.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
Agent.fit(env, nb_steps=10000, visualize=True, verbose=2)

# Save the trained model
Agent.save_weights('Agent_weights.h5f', overwrite=True)

# Evaluate the trained agent
Agent.test(env, nb_episodes=5, visualize=True)
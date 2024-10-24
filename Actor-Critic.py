import gymnasium as gym
import numpy as np
import tqdm
import statistics
import collections
import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import os



import math



env = gym.make('LunarLander-v3', render_mode="human")
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
	def __init__(self, num_actions: int, num_hidden_units: int):
		super().__init__()

		self.common = layers.Dense(num_hidden_units, activation='relu')
		self.actor = layers.Dense(num_actions)
		self.critic = layers.Dense(1)

	def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		x = self.common(inputs)
		return self.actor(x), self.critic(x)


num_actions = env.action_space.n
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	state, reward, done, _, _= env.step(action)

	return (state.astype(np.float32),
		np.array(reward, np.int32),
		np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
	return tf.numpy_function(env_step, [action],
				[tf.float32, tf.int32, tf.int32])

def run_episode(
	initial_state: tf.Tensor,
	model: tf.keras.Model,
	max_steps: int,
	training: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
	
	action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

	initial_state_shape = initial_state.shape
	state = initial_state

	for t in tf.range(max_steps):
		if (training == False):
			env.render()

		state = tf.expand_dims(state, 0)
		action_logits_t, value = model(state)

		action = tf.random.categorical(action_logits_t, 1)[0, 0]
		action_probs_t = tf.nn.softmax(action_logits_t)

		values = values.write(t, tf.squeeze(value))

		action_probs = action_probs.write(t, action_probs_t[0, action])

		state, reward, done = tf_env_step(action)
		state.set_shape(initial_state_shape)

		rewards = rewards.write(t, reward)

		if reward == 100:
			print('================== WON')
		elif reward == -100:
			print('================== DESTROYED')
		if tf.cast(done, tf.bool):
			break

	action_probs = action_probs.stack()
	values = values.stack()
	rewards = rewards.stack()

	print(f'Lasted : {t} steps')

	return action_probs, values, rewards


def get_expected_return(
	rewards: tf.Tensor,
	gamma: float,
	standarize: bool = True) -> tf.Tensor:
	
	n = tf.shape(rewards)[0]
	returns = tf.TensorArray(dtype=tf.float32, size=n)

	rewards = tf.cast(rewards[::-1], dtype=tf.float32)
	discounted_sum = tf.constant(0.0)
	discounted_sum_shape = discounted_sum.shape

	for i in tf.range(n):
		reward = rewards[i]
		discounted_sum = reward + gamma*discounted_sum
		discounted_sum.set_shape(discounted_sum_shape)
		returns = returns.write(i, discounted_sum)
	returns = returns.stack()[::-1]

	if standarize:
		returns = ((returns - tf.math.reduce_mean(returns)) / 
					(tf.math.reduce_std(returns) + eps))

	return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
	action_probs: tf.Tensor,
	values: tf.Tensor,
	returns: tf.Tensor) -> tf.Tensor:
	advantage = returns - values

	action_log_probs = tf.math.log(action_probs)
	actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

	critic_loss = huber_loss(values, returns)
	return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function
def train_step(
	initial_state: tf.Tensor,
	model: tf.keras.Model,
	optimizer: tf.keras.optimizers.Optimizer,
	gamma: float,
	max_steps_per_episode: int) -> tf.Tensor:

	with tf.GradientTape() as tape:
		action_probs, values, rewards = run_episode(
			initial_state, model, max_steps_per_episode)

		returns = get_expected_return(rewards, gamma)

		action_probs, values, returns = [
			tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

		loss = compute_loss(action_probs, values, returns)

	grads = tape.gradient(loss, model.trainable_variables)

	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	episode_reward = tf.math.reduce_sum(rewards)

	return episode_reward



min_episodes_criterion = 300
max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 180
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

train = False

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)



if (train == True):
	best_reward = -math.inf
	with tqdm.trange(max_episodes) as t:
		for i in t:
			initial_state = tf.constant(env.reset(), dtype=tf.float32)

			episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

			episodes_reward.append(episode_reward)
			running_reward = statistics.mean(episodes_reward)

			t.set_description(f'Episode {i}')
			t.set_postfix(
				episode_reward=episode_reward, running_reward=running_reward)

			if running_reward >= best_reward:
				best_reward = running_reward
				model.save_weights(os.path.join("tmp2/lunar_lander", '_ac'))

			if running_reward >= reward_threshold and i >= min_episodes_criterion:
				break

	#model.save_weights(os.path.join("tmp2/lunar_lander", '_ac'))
	print(f'\n Solved at episode {i}: average reward: {running_reward}')
else:
	model.load_weights(os.path.join("tmp2/lunar_lander", '_ac'))
	for i in range(10):
		
		initial_state = tf.constant(env.reset()[0], dtype=tf.float32)
		
		action_probs, values, rewards = run_episode(
			initial_state, model, max_steps_per_episode, training=False)

		print(f'Finished game {i}; Average_reward: {tf.math.reduce_sum(rewards)}')







#%%
"""
from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display


display = Display(visible=0, size=(400, 300))
display.start()


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
  screen = env.render()
  im = Image.fromarray(screen)

  images = [im]
  
  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))
  
    if done:
      break
  
  return images

print('Creating GIF')
# Save GIF image
images = render_episode(env, model, max_steps_per_episode)
image_file = 'cartpole-v0.gif'
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(
    image_file, save_all=True, append_images=images[1:], loop=0, duration=1)"""
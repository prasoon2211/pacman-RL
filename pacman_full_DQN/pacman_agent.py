import gym
import time
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque
from skimage import color
from skimage.transform import resize
from baselines import deepq
from baselines.common.misc_util import relatively_safe_pickle_dump, pickle_load
from baselines.common import tf_util
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines import logger
from baselines.deepq.utils import Uint8Input, load_state, save_state

env = gym.make("MsPacman-v0")
input_height, input_width = (86, 80)
batch_size = 32
update_freq = 10000
learn_freq = 4
save_freq = 500000
action_space_size = env.action_space.n
NUM_STEPS = 20000000
replay_memory_size = 10000
replay_alpha = 0.6
replay_beta = 0.4
replay_epsilon = 1e-6
is_load_model = True
watch_flag = False
fps = 30 #frames shown per second when watch_flag == True

log_csv_writer = logger.make_output_format("csv", "logs")
log = logger.Logger("logs", [log_csv_writer])

def preprocess_frame(frame):
    """Given a frame, scales it and converts to grayscale"""
    im = resize(color.rgb2gray(frame)[:176, :], (input_height, input_width), mode='constant')
    return im

"""
model arch
----------

Conv1 (8x8x32 filter) -> ReLU -> Conv2 (4x4x64 filter) -> ReLU -> Conv3 (3x3x64 filter) -> ReLU ->

Here the model bifurcates into two.
----------------------------------
state value piece
-----------------
FC4 (512 neurons) -> ReLU -> Batch Norm -> FC5 (1 neuron) -> Output state score

action score piece
------------------
FC4' (512 neurons) -> ReLU -> Batch Norm -> FC5' (9 neurons) -> ReLU -> Output Q-value for each action

"""
def q_function_nn(obs, action_space_size, scope, reuse=False):
    """
    This nn calculates Q-values for all actions in a state
    This function returns the nn computation graph.
    obs: The state encoded as a tesnor
    action_space_size: Total possible actions
    scope: TF variable scope name
    reuse: bool. Definition large; see in TF docs

    Here we are using the dueling model described in the paper https://arxiv.org/pdf/1511.06581.pdf
    Dueling Network Architectures for Deep Reinforcement Learning.

    Essentially, we split the neural network after the convolutional layers into two parts,
    one calculates the state score and the other calculates the "state-dependent action advantage function"
    which is essentially Q-values for all actions given the fixed state.

    This model performs better than the single value output model.
    The model representation in tensorflow is used from baselines DQN model.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("convolutional_layers"):
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            state_hidden = tf.nn.relu(state_hidden)
            state_hidden = layers.batch_norm(state_hidden)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            actions_hidden = tf.nn.relu(actions_hidden)
            actions_hidden = layers.batch_norm(actions_hidden)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=action_space_size, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return state_score + action_scores

def save_model(dict_state):
    save_state("saved_model/model.ckpt")
    relatively_safe_pickle_dump(dict_state, "saved_model/model_state.pkl.zip", compression=True)

def load_model():
    load_state("saved_model/model.ckpt")
    dict_state = pickle_load("saved_model/model_state.pkl.zip", compression=True)
    return dict_state

def build_state(state_deque):
    """Stack frames together to construct state"""
    return np.array(list(state_deque))

def main():
    with tf_util.make_session() as session:
        act_fn, train_fn, target_update_fn, debug_fn = deepq.build_train(
            make_obs_ph=lambda name: Uint8Input([4, input_height, input_width], name=name),
            q_func=q_function_nn,
            num_actions=action_space_size,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=True)
        epsilon = PiecewiseSchedule([(0, 1.0),
                                     (10000, 1.0), # since we start training at 10000 steps
                                     (20000, 0.4),
                                     (50000, 0.2),
                                     (100000, 0.1),
                                     (500000, 0.05)], outside_value=0.01)
        replay_memory = PrioritizedReplayBuffer(replay_memory_size, replay_alpha)
        beta = LinearSchedule(int(NUM_STEPS/4), initial_p=replay_beta, final_p=1.0)
        tf_util.initialize()
        target_update_fn()

        state = preprocess_frame(env.reset())
        state_deque = deque([state]*4, maxlen=4) # deque where all incoming frames get stored
        state_full = build_state(state_deque) # stacks 4 frames together
        dq = [] # a queue to store episode rewards
        start_step = 1
        episode = 1
        if is_load_model:
            dict_state = load_model()
            replay_memory = dict_state["replay_memory"]
            dq = dict_state["dq"]
            start_step = dict_state["step"] + 1

        for step in itertools.count(start=start_step):
            action = act_fn(state_full[np.newaxis], update_eps=epsilon.value(step))[0]
            state_tplus1, reward, is_finished, _ = env.step(action)
            state_tplus1 = preprocess_frame(state_tplus1)
            state_deque.append(state_tplus1)
            state_tplus1_full = build_state(state_deque)
            dq.append(reward)
            if watch_flag:
                env.render()
                time.sleep(1.0/fps)
            replay_memory.add(state_full, action, reward, state_tplus1_full, float(is_finished))
            state_full = state_tplus1_full
            if is_finished:
                ep_reward = sum(dq)
                log.logkv("Steps", step)
                log.logkv("Episode reward", ep_reward)
                log.logkv("Episode number", episode)
                log.dumpkvs()
                print("Step ", step, ". Finished episode ", episode, " with reward ", ep_reward)
                dq = []
                state = preprocess_frame(env.reset())
                state_deque = deque([state]*4, maxlen=4)
                state_full = build_state(state_deque)
                episode += 1
                for _ in range(30):
                    # NOOP for ~90 frames to skip the start screen. Range 30 used because each
                    # step executed for 3 frames on average. Action 0 stands for no-operation
                    env.step(0)
                    if watch_flag:
                        env.render()

            if step > 10000 and step % learn_freq == 0:
                # only start training after 10000 steps are completed
                batch = replay_memory.sample(batch_size, beta=beta.value(step))
                states = batch[0]
                actions = batch[1]
                rewards = batch[2]
                states_tplus1 = batch[3]
                finished_vars = batch[4]
                weights = batch[5]
                state_indeces = batch[6]
                errors = train_fn(states, actions, rewards, states_tplus1, finished_vars, weights)
                priority_order_new = np.abs(errors) + replay_epsilon
                replay_memory.update_priorities(state_indeces, priority_order_new)

            if step % update_freq == 0:
                target_update_fn()

            if step % save_freq == 0:
                print("State save", step)
                dict_state = {
                    "step": step,
                    "replay_memory": replay_memory,
                    "dq": dq
                }
                save_model(dict_state)

            if step > NUM_STEPS:
                print("Finished training. Saving model to ./saved_model/model.ckpt")
                dict_state = {
                    "step": step,
                    "replay_memory": replay_memory,
                    "dq": dq
                }
                save_model(dict_state)
                break
main()

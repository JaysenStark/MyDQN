from constants import *
from dqn import DQN
import numpy as np
import gym
from random import random
from random import sample
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

class Agent(object):
    """docstring for Agent"""
    def __init__(self, env, dqn):
        self.dqn = dqn
        self.Q = dqn.generate_model()
        self.Q_ = dqn.copy_model(self.Q)
        self.exploration_factor = INITIAL_EXPLORATION
        
        self.env = env
        self.steps = 0
        self.memory = deque()
        self.train_minibatches = []


    def choose_action(self, sequence):
        prediction = self.Q.predict(np.asarray([np.stack(sequence, -1)]))[0]
        return np.argmax(prediction)

    '''return random action or action with best predicted Q value'''
    def generate_action(self, sequence):
        if random() <= self.exploration_factor:
            return self.env.action_space.sample()
        return self.choose_action(sequence)

    '''add transition in self.memory util size limit reached, 
       observation need to be preprocessed before sotrage'''
    def initialize_replay_memory(self, render=False):
        # memory size limit not reached, continuing
        while len(self.memory) < REPLAY_MEMORY_SIZE:
            observation = self.env.reset()
            observation = self.preprocess_observation(observation)
            sequence = [observation] * AGENT_HISTORY_LENGTH
            done = False
            action = -1

            while not done and len(self.memory) < REPLAY_MEMORY_SIZE:
                if render:
                    self.env.render()
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                observation = self.preprocess_observation(observation)
                sequence = sequence[1:]
                sequence.append(observation)
                assert not type(observation) == float
                self.memory.append(Transition(sequence, action, reward, observation, done))
        print("memory initialization done! with memory size: %d" % len(self.memory))


    def train(self):
        # minibatch is a list which contains transitions
        for minibatch in self.train_minibatches:
            sequences = []
            next_sequences = []
            for transition in minibatch:
                assert not type(transition.observation) == float
                sequences.append(np.stack(transition.sequence, -1))
                next_sequence = transition.sequence[1:] + [transition.observation]
                try:
                    next_sequences.append(np.stack(next_sequence, -1))
                except:
                    for i in range(4):
                        print(i)
                        print(next_sequence[i].shape)

            targets = self.Q.predict(np.asarray(sequences))
            new_sequences_predictions = self.Q_.predict(np.asarray(next_sequences))

            for index, prediction, in enumerate(new_sequences_predictions):
                done = minibatch[index].done
                reward = minibatch[index].done
                action = minibatch[index].action
                max_q = np.max(prediction) if not done else 0
                targets[index][action] = reward + DISCOUNT_FACTOR * max_q

            print(self.Q.train_on_batch(np.asarray(sequences), targets))


    def run(self, episodes, render=False, do_train=True):
        for episode in range(episodes):
            observation = self.env.reset()
            observation = self.preprocess_observation(observation)
            sequence = [observation] * AGENT_HISTORY_LENGTH
            done = False

            action = -1
            while not done:
                if render:
                    self.env.render()
                # do action every four frame
                if self.steps % ACTION_REPEAT == 0:
                    action = self.generate_action(sequence)

                observation, reward, done, info = self.env.step(action)
                observation = self.preprocess_observation(observation)
                self.memory.append(Transition(sequence, action, reward, observation, done))
                self.memory.popleft()
                sequence = sequence[1:]
                sequence.append(observation)

                self.steps += 1
                self.exploration_factor = max(FINAL_EXPLORATION, self.exploration_factor - 1.0/ FINAL_EXPLORATION_FRAME)

                self.train_minibatches.append(sample(self.memory, MINIBATCH_SIZE))
                # train when  len(self.train_minibatches) >= self.UPDATE_FREQUENCY and do_tranin == True
                if len(self.train_minibatches) >= UPDATE_FREQUENCY and do_train:
                    self.train()
                    self.train_minibatches = []

                if self.steps % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    self.Q_ = self.dqn.copy_model(self.Q)
                    print("############", self.exploration_factor)





    '''convert original image to gray image, then resize to '''
    def preprocess_observation(self, observation):
        gray_observation = rgb2gray(observation)
        processed_observation = np.uint8(resize(gray_observation, (IMG_WIDTH, IMG_HEIGHT), mode="reflect") * 255)
        return np.reshape(processed_observation, (IMG_WIDTH, IMG_HEIGHT))


class Transition(object):

    def __init__(self, sequence, action, reward, observation, done):
        self.sequence = sequence
        self.action = action
        self.reward = reward
        self.observation = observation
        self.done = done


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    agent = Agent(env, DQN())
    agent.initialize_replay_memory(False)
    agent.run(100000)
    print("end")




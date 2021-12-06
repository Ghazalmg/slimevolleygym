import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
#from replay_buffer import ReplayBuffer
import slimevolleygym
import numpy as np

BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'SlimeVolley-v0'
PRINT_INTERVAL = 1

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = 2**env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

model2 = MyModel(state_shape, n_actions).to(device)
target2 = MyModel(state_shape, n_actions).to(device)
target2.load_state_dict(model.state_dict())
target2.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# memory = ReplayBuffer()
def binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def choose_action(state, test_mode=False):
    r = np.random.random()
    if r<EPS_EXPLORATION:
        action1 = torch.tensor(env.action_space.sample())
    else:
        action1 = binary(torch.argmax(model.forward(state)),3)
    
    r = np.random.random()
    if r<EPS_EXPLORATION:
        action2 = torch.tensor(env.action_space.sample())
    else:
        action2 = binary(torch.argmax(model2.forward(state)),3)
    return action1,action2
   

def optimize_model(state, action1, action2, next_state, reward, done):
    with torch.no_grad():
        if done:
            y1 =torch.tensor(reward)
            y2 =torch.tensor(-reward)
        else:
            y1 = reward + GAMMA * torch.max(target(torch.tensor(next_state)))
            y2 = -reward + GAMMA * torch.max(target(torch.tensor(next_state)))
    criterion = nn.MSELoss()
    loss1 = criterion(torch.argmax(model.forward(state)),y1)    
    loss2 = criterion(torch.argmax(model2.forward(state)),y2)    
    optimizer.zero_grad()
    loss1.backward()
    loss2.backward()
    optimizer.step()
    
def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score1 = -float("inf")
    best_score2 = -float("inf")

    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward1 = 0
        episode_total_reward2 = 0
        state = env.reset()
        for t in count():
            action1,action2 = choose_action(state)
            next_state, reward, done, _ = env.step(action1,action2)
            steps_done += 1
            episode_total_reward1 += reward
            episode_total_reward2 -= reward

            optimize_model(state, action1, action2, next_state, reward, done)

            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())
            target2.load_state_dict(model2.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score1,score2 = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score1 > best_score1:
                best_score1 = score1
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)
        if score2 > best_score2:
                best_score2 = score2
                torch.save(model2.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
                print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
                print('-'*10)    


train_reinforcement_learning()
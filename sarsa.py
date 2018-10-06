from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
from time import sleep
import random

game = FlappyBird(pipe_gap = 323)

p = PLE(game, fps=30, display_screen=True, force_fps=False,
                reward_values= {
                    "positive": 1.0,
                    # "negative": -1.0,
                    "tick": 0.0,
                    "loss": -1000.0,
                    "win": 5.0
                }
            )

p.init()
# reward = 0.0

class NaiveAgent():

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

    def createStateActionPolicy(self,game):
        Q = {}
        for state in range(-15, int(game.height*0.79)):
            Q[state] = {0: 0, 1: 0}
        return Q

    def __argmax_Q(self,Q, s):
        Q_list = list(map(lambda x: x[1], Q[s].items()))  # 13.
        indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
        max_Q = random.choice(indices)
        return max_Q

    def greedy_policy(self,Q):
        policy = {}
        for state in Q.keys():
            policy[state] = self.__argmax_Q(Q, state)
        return policy

    def getAction(self,a):
        return self.actions[a]

    def myAction(self,obs):
        return self.actions[np.random.randint(0, len(self.actions))]

myAgent = NaiveAgent(p.getActionSet())
Q = myAgent.createStateActionPolicy(game)

# print(Q)

episodes = 100
step_size = 0.01
exploration_rate = 0.01

def sarsa():

    gp = None

    for episode in range(episodes):

        p.reset_game()
        obs = game.getGameState()
        S = obs['player_y']

        A = myAgent.greedy_policy(Q)[S]  # 4. Deciding on first action
        Action = myAgent.getAction(A)

        cumulative_reward = 0

        while(not p.game_over()):

            reward = p.act(Action)
            cumulative_reward = cumulative_reward + reward + 1
            print(cumulative_reward)
            Obs_prime = game.getGameState()
            S_prime = Obs_prime['player_y']

            # print(A)

            A_prime = myAgent.greedy_policy(Q)[S_prime]
            # A_prime = 0

            # cumulative_reward = cumulative_reward + reward
            Q[S][A] = Q[S][A] + step_size * (reward + exploration_rate * Q[S_prime][A_prime] - Q[S][A])

            # td_target = reward + discount_factor * Q[next_state][next_action]
            # td_delta = td_target - Q[state][action]
            # Q[state][action] += alpha * td_delta

            S = S_prime
            A = A_prime
            Action = myAgent.getAction(A_prime)

            gp = myAgent.greedy_policy(Q)

        print(cumulative_reward)
        print(gp)
    return gp, Q

sarsa()
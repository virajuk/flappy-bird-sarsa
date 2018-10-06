from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
from time import sleep
import random

# game = FlappyBird(pipe_gap = 150)
game = FlappyBird()

# print(game.pipe_gap)

p = PLE(game, fps=30, display_screen=True, force_fps=False,
                reward_values= {
                    "positive": 1.0,
                    # "negative": -1.0,
                    # "tick": 0.0,
                    "loss": -1000.0,
                    # "win": 5.0
                }
            )

p.init()
# reward = 0.0

class NaiveAgent():

    def __init__(self, actions):
        self.actions = actions
        self.state_space = {}

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

    def setTrainedQTable(self,Q):

        self.state_space = Q
        # return self

    def createStateActionPolicy(self,game):
        Q = {}
        self.__createStateDict()
        # for state in range(-15, int(game.height*0.79)):
        # for state in range(-400, 200):
        for state in self.state_space.keys():
            Q[state] = {0: 0, 1: 0}
        return Q

    def __createStateDict(self):

        state = {}
        state_key = 0

        for player_tan in range(-330, 440):
            state_key += 1
            state[state_key] = { 'player_tan' : player_tan }

        self.state_space = state
        # print(state)
        return state

    def getStateFromStateSpace(self,obs):

        verticle_distance = int(obs['next_pipe_bottom_y'] - (game.pipe_gap)/2 - obs['player_y'])
        next_pipe_dist_to_player = int(obs['next_pipe_dist_to_player'])


        if (next_pipe_dist_to_player > 1) and verticle_distance > 0:
            player_tan = int(verticle_distance/next_pipe_dist_to_player)
        else:
            player_tan = 440

        if (next_pipe_dist_to_player > 1) and verticle_distance < 0:
            player_tan = int(verticle_distance / next_pipe_dist_to_player)
        else:
            player_tan = -330

        current_state = { 'player_tan' : player_tan }

        for state in self.state_space:

            if(current_state == self.state_space[state]):
                return state


    def __argmax_Q(self,Q, s):

        Q_list = list(map(lambda x: x[1], Q[s].items()))
        indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
        max_Q = random.choice(indices)
        return max_Q

    def greedy_policy(self,Q):

        policy = {}
        for state in Q.keys():
            policy[state] = self.__argmax_Q(Q, state)
        return policy

    def getPolicy(self,Q):
        pass

    def getAction(self,a):
        return self.actions[a]

    def myAction(self,obs):
        return self.actions[np.random.randint(0, len(self.actions))]

myAgent = NaiveAgent(p.getActionSet())
Q = myAgent.createStateActionPolicy(game)

# print(Q)

# myAgent.setTrainedQTable(Q)

starting_episode = 0
episodes = 5000
# episodes = 4

# step_size = 0.01
# exploration_rate = 0.01

discount_factor = 0.9
alpha = 0.1

obs = game.getGameState()
# print(obs)

def sarsa():

    # gp = None
    # max_reward = -1000

    # with open("output.txt", "a") as file:

    for episode in range(starting_episode,episodes):

        p.reset_game()
        obs = game.getGameState()
        # S = obs['player_y']

        S = myAgent.getStateFromStateSpace(obs)
        # print(state)
        # print(Q)

        # print(obs)
        # print(S)

        # S = (obs['player_y'] - obs['next_pipe_bottom_y'])
        # # obs['player_y'] > (obs['next_pipe_bottom_y']

        A = myAgent.greedy_policy(Q)[S]  # 4. Deciding on first action

        # print(A)

        Action = myAgent.getAction(A)

        # print(Action)

        cumulative_reward = 0

        while(not p.game_over()):

            reward = p.act(Action)
            cumulative_reward = cumulative_reward + reward
            bird_alive_reward = reward + 1

            Obs_prime = game.getGameState()
            S_prime = myAgent.getStateFromStateSpace(Obs_prime)

            A_prime = myAgent.greedy_policy(Q)[S_prime]

            Target = bird_alive_reward + discount_factor * Q[S_prime][A_prime]
            Delta = Target - Q[S][A]
            Q[S][A] = Q[S][A] + alpha*Delta

            S = S_prime
            A = A_prime
            Action = myAgent.getAction(A_prime)

        # max_reward = max(max_reward, cumulative_reward)

        with open("sarsa011-output.txt", "a") as file:

            file.write("[" + str(episode) +"," + str(cumulative_reward + 1000) + "],\n")

        with open("sarsa011-q-table.txt", "w") as file:

            file.write(str(Q))

        # print('Episode = ' + str(episode) + ', Reward = ' + str(cumulative_reward) + ', Max Reward = ' + str(max_reward))
        # print(gp)
        # print(Obs_prime)
        # print(Q)

    # return gp, Q

sarsa()
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
        # state_key = 0

        # for verticle_distance in range(-62, 82):
        #     for horizontal_distance in range(0, 64):
        #         state_key = str(verticle_distance) + '_' + str(horizontal_distance)
        #         # state[state_key] = { 'player_tan' : player_tan, 'player_tan_top' : player_tan_top }
        #         state[state_key] = { 'verticle_distance' : verticle_distance, 'horizontal_distance': horizontal_distance }


        #player_vel
        for player_tan in range(-250, 300):
            for player_vel in range(-7, 7):
                state_key = str(player_tan) + '_' + str(player_vel)
                state[state_key] = {'player_tan': player_tan, 'player_vel' : player_vel }
        self.state_space = state

        return state

    def getStateFromStateSpace(self,obs):

        # verticle_distance = int(obs['next_pipe_bottom_y'] - obs['player_y'])/5
        # horizontal_distance = int(obs['next_pipe_dist_to_player'])/5
        #
        # current_state = str(verticle_distance) + '_' + str(horizontal_distance)
        # return current_state

        verticle_distance = int(obs['next_pipe_bottom_y'] - (game.pipe_gap) / 2 - obs['player_y'])
        next_pipe_dist_to_player = int(obs['next_pipe_dist_to_player'])
        player_vel = int(obs['player_vel']/3)

        if(verticle_distance > 299):
            verticle_distance = 299

        if (next_pipe_dist_to_player > 1) and verticle_distance > 0:
            player_tan = int(verticle_distance/next_pipe_dist_to_player)
        else:
            player_tan = 299

        if (verticle_distance < -250):
            verticle_distance = -250

        if (next_pipe_dist_to_player > 1) and verticle_distance < 0:
            player_tan = int(verticle_distance / next_pipe_dist_to_player)
        else:
            player_tan = -250

        current_state = str(player_tan) + '_' + str(player_vel)
        return current_state


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

starting_episode = 1004
episodes = 5000
# episodes = 1

alpha = 0.1
discount_factor = 0.9


obs = game.getGameState()
# print(obs)

def sarsa():

    # gp = None
    max_reward = -1000

    # with open("output.txt", "a") as file:

    gp = None

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

            # game.showScore()

            reward = p.act(Action)
            # reward += 1
            cumulative_reward = cumulative_reward + reward
            bird_alive_reward = reward + 1

            Obs_prime = game.getGameState()
            # print(Obs_prime)
            S_prime = myAgent.getStateFromStateSpace(Obs_prime)

            A_prime = myAgent.greedy_policy(Q)[S_prime]

            Target = bird_alive_reward + discount_factor * Q[S_prime][A_prime]
            Delta = Target - Q[S][A]
            Q[S][A] = Q[S][A] + alpha*Delta

            S = S_prime
            A = A_prime
            Action = myAgent.getAction(A_prime)

            gp = myAgent.greedy_policy(Q)

            # print(game.score)

        with open("output-052.txt", "a") as file:

            # if (episode == 399):
            #     cumulative_reward += 3500.0

            file.write("[ " + str(episode) +", " + str(cumulative_reward + 1000) + "],\n")

        with open("q-table-052.txt", "w") as file:

            file.write(str(Q))





        # print('Episode = ' + str(episode) + ', Reward = ' + str(cumulative_reward) + ', Max Reward = ' + str(max_reward))
        # print(gp)
        # print(Obs_prime)
        # print(Q)

    return gp, Q

sarsa()

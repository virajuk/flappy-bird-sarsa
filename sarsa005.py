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
        self.state_space = {}

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

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

        for player_tan in range(-33, 44):
            # for player_tan_top in range(-43, 36):
                state_key += 1
                # state[state_key] = { 'player_tan' : player_tan, 'player_tan_top' : player_tan_top }
                state[state_key] = { 'player_tan' : player_tan }

        self.state_space = state
        # print(state)
        return state

    def getStateFromStateSpace(self,obs):

        # print(obs['next_pipe_bottom_y'])
        # print(obs['player_y'])
        # print(obs['next_pipe_dist_to_player'])

        # verticle_distance = int((obs['next_pipe_bottom_y'] - (obs['player_y'] + 40))/8)
        verticle_distance = int(obs['next_pipe_bottom_y'] - (game.pipe_gap)/2 - obs['player_y'])
        # verticle_distance = int(obs['next_pipe_bottom_y'] - obs['player_y'])/2
        # next_pipe_dist_to_player = int((obs['next_pipe_dist_to_player'])/10)
        # verticle_distance_upper_pipe = int(obs['next_pipe_top_y'] + (game.pipe_gap)/2 - obs['player_y'])
        # player_vel = int(obs['player_vel']/2)

        next_pipe_dist_to_player = int(obs['next_pipe_dist_to_player'])

        # next_next_pipe_dist_to_player = int(obs['next_next_pipe_dist_to_player'])



        # if (next_pipe_dist_to_player > 10) and verticle_distance > 0 :
        #     player_tan = int(verticle_distance/next_pipe_dist_to_player)
        #     # print(player_tan)
        # else:
        #     # player_tan = 43
        #     player_tan = int(verticle_distance / next_next_pipe_dist_to_player)
        #
        # if (next_pipe_dist_to_player > 10) and verticle_distance < 0:
        #     player_tan = int(verticle_distance / next_pipe_dist_to_player)
        #     # print(player_tan)
        # else:
        #     player_tan = int(verticle_distance / next_next_pipe_dist_to_player)




        if (next_pipe_dist_to_player > 10) and verticle_distance > 0:
            player_tan = int(verticle_distance/next_pipe_dist_to_player)
            # player_tan_top = int(verticle_distance_upper_pipe / next_pipe_dist_to_player)
        else:
            player_tan = 43
            # player_tan_top = -32
            # player_tan = int(verticle_distance / next_next_pipe_dist_to_player)
            # player_tan_top = int(verticle_distance_upper_pipe / next_next_pipe_dist_to_player)

        if (next_pipe_dist_to_player > 10) and verticle_distance < 0:
            player_tan = int(verticle_distance / next_pipe_dist_to_player)
            # print(player_tan)
        else:
            player_tan = -32



        # if (next_pipe_dist_to_player > 10) and verticle_distance_upper_pipe < 0:
        #     player_tan_top = int(verticle_distance_upper_pipe / next_pipe_dist_to_player)
        #     # print(player_tan)
        # else:
        #     # player_tan_top = int(verticle_distance_upper_pipe / next_next_pipe_dist_to_player)
        #     player_tan_top = 35
        #
        # if (next_pipe_dist_to_player > 10) and verticle_distance_upper_pipe > 0:
        #     player_tan_top = int(verticle_distance_upper_pipe / next_pipe_dist_to_player)
        #     # print(player_tan)
        # else:
        #     # player_tan_top = int(verticle_distance_upper_pipe / next_next_pipe_dist_to_player)
        #     player_tan_top = -42


        # if (next_pipe_dist_to_player > 10):
        #     player_tan_top = int(verticle_distance_upper_pipe / next_pipe_dist_to_player)
        # else:
        #     player_tan_top = int(verticle_distance_upper_pipe / next_next_pipe_dist_to_player)



        current_state = { 'player_tan': player_tan }
        # current_state = { 'player_tan': player_tan, 'player_tan_top' : player_tan_top }

        # print(current_state)
        # print(self.state_space)
        # print(player_vel)

        for state in self.state_space:
            # print(state)
            # print(self.state_space[state])
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

episodes = 2000
# episodes = 4

step_size = 0.01
exploration_rate = 0.01

discount_factor = 1.0
alpha = 0.5

obs = game.getGameState()
# print(obs)

def sarsa():

    # gp = None
    max_reward = -1000

    for episode in range(episodes):

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
        #
        A = myAgent.greedy_policy(Q)[S]  # 4. Deciding on first action
        Action = myAgent.getAction(A)

        # print(Action)

        cumulative_reward = 0

        while(not p.game_over()):

            reward = p.act(Action)
            #reward += 1
            cumulative_reward = cumulative_reward + reward

            # print(cumulative_reward)
            Obs_prime = game.getGameState()
            # print(Obs_prime)
            # S_prime = Obs_prime['player_y'] - Obs_prime['next_pipe_bottom_y']
            S_prime = myAgent.getStateFromStateSpace(Obs_prime)


            # print(S_prime)
            # print(A)

            A_prime = myAgent.greedy_policy(Q)[S_prime]
            # print(S_prime)
            # A_prime = 0

            # cumulative_reward = cumulative_reward + reward
            # Q[S][A] = Q[S][A] + step_size * (reward + exploration_rate * Q[S_prime][A_prime] - Q[S][A])

            Target = cumulative_reward + discount_factor * Q[S_prime][A_prime]
            Delta = Target - Q[S][A]
            Q[S][A] = Q[S][A] + alpha*Delta

            # td_target = reward + discount_factor * Q[next_state][next_action]
            # td_delta = td_target - Q[state][action]
            # Q[state][action] += alpha * td_delta

            S = S_prime
            A = A_prime
            Action = myAgent.getAction(A_prime)

            # gp = myAgent.greedy_policy(Q)

            # sleep(0.015)
        # print(obs['next_pipe_bottom_y'])
        # print(obs['player_y'])
        #
        # print(S)
        # print(cumulative_reward)

        max_reward = max(max_reward, cumulative_reward)

        print('Episode = ' + str(episode) + ', Max Reward = ' + str(max_reward))
        # print(gp)
        # print(Obs_prime)
        # print(Q)

    # return gp, Q

sarsa()
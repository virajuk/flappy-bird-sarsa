from ple.games.flappybird import FlappyBird
import numpy as np
from ple import PLE
from time import sleep

game = FlappyBird(pipe_gap = 110)

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

class NaiveAgent():

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):

        return self.actions[np.random.randint(0, len(self.actions))]
        # return self.actions[1]

    def getActionCount(self):
        return len(self.actions)

    def myAction(self,obs):


        # print(obs['next_pipe_bottom_y'])
        # print(obs['player_y'])
        # print(obs['next_pipe_top_y'])

        # print(obs)

        if(obs['player_y'] > (obs['next_pipe_bottom_y'] - 55)):
            # print str(119) + ' player_y = ' + str(obs['player_y']) + ' next_pipe_top_y = ' + str(obs['next_pipe_top_y']) + ' next_next_pipe_top_y = ' + str(obs['next_next_pipe_top_y'])
            # print str(119) + ' player_y = ' + str(obs['player_y']) + ' next_pipe_top_y = ' + str(obs['next_pipe_top_y']) + ' next_pipe_bottom_y = ' + str(obs['next_pipe_bottom_y'])
            return self.actions[0]
            # return self.actions[np.random.randint(0, len(self.actions))]
        else:
            # print str(None) + ' player_y = ' + str(obs['player_y']) + ' next_pipe_top_y = ' + str(obs['next_pipe_top_y']) + ' next_pipe_bottom_y = ' + str(obs['next_pipe_bottom_y'])
            return self.actions[1]
            # return self.actions[np.random.randint(0, len(self.actions))]

        # if (obs['player_y'] < obs['next_pipe_top_y']):
        #     # return 119
        #     print False

        # return self.actions
        # print(self.actions[np.random.randint(0, len(self.actions))])

        # action = np.random.randint(0, len(self.actions))
        # print(action)

        # return self.actions[action]

myAgent = NaiveAgent(p.getActionSet())

nb_frames = 1000
reward = 0.0

# print(game.getActions())
# print(game.actions)

# print(myAgent.getActionCount())

print(game.height)

# for f in range(nb_frames):
#
#     if p.game_over():  # check if the game is over
#         # reward = reward - 1000
#         p.reset_game()
#         # break
#
#     # obs = p.getScreenRGB()
#     obs = game.getGameState()
#     # action = myAgent.myAction(obs)
#     action = myAgent.pickAction(reward,obs)
#     # print(action)
#
#     # print(p.getGameState())
#     # print(game.getGameState())
#     # next_next_pipe_top_y, player_vel, next_pipe_bottom_y, next_pipe_top_y, next_next_pipe_bottom_y, next_pipe_dist_to_player, next_next_pipe_dist_to_player, player_y = game.getGameState()
#
#     # print(next_next_pipe_top_y)
#     # print(game.getGameState()["next_next_pipe_top_y"])
#     print(obs["player_y"])
#
#     reward =  reward + p.act(action)
#
#     # print(obs)
#     sleep(0.01)
#     # print(action)
#     print(reward)

while(not p.game_over()):

    # print('here')
    obs = game.getGameState()
    print(obs)
    # print(obs["player_y"])
    action = myAgent.pickAction(reward,obs)
    reward = reward + p.act(action)
    print(reward)
import random
# import gym.utils as utils
import gym_utils as utils
from IPython.display import clear_output
from time import sleep
import numpy as np
from random import randint
import gym


def run_game(env, policy, display=True, should_return=True):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s

        if display:
            clear_output(True)
            env.render()
            sleep(0.5)

        timestep = []
        timestep.append(s)
        action = policy[s]
        state, reward, finished, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    if display:
        clear_output(True)
        env.render()
        sleep(0.5)

    if should_return:
        return episode

def argmax_Q(Q, s):
    Q_list = list(map(lambda x: x[1], Q[s].items())) # 13.
    indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
    max_Q = random.choice(indices)
    return max_Q

def greedy_policy(Q):
    policy = {}
    for state in Q.keys():
        policy[state] = argmax_Q(Q, state)
    return policy

def field_list(env):
    l = []
    for i in range(4):
        for row in env.env.desc[i]:
        # for row in list(map(lambda x: list([str(y)[-2] for y in x]), list(env.env.desc))):
            for field in row:
                l.append(field)
    return l

def create_state_action_dictionary(env, policy):
    Q = {}
    fields = field_list(env)
    # print(fields)
    for key in policy.keys():
        if fields[key] in ['F', 'S']:
            Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
        else:
            Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q

# def test_policy(policy, env):
#     wins = 0
#     r = 1000
#     for i in range(r):
#         w = run_game(env, policy, display=False)[-1][-1]
#         if w == 1:
#             wins += 1
#     return wins / r


def sarsa(env, episodes=12000, step_size=0.01, exploration_rate=0.01):
    policy = utils.create_random_policy(env)  # Create policy, just for the util function to create Q
    Q = create_state_action_dictionary(env, policy)  # 1. Initialize value dictionary formated: { S1: { A1: 0.0, A2: 0.0, ...}, ...}
    gp = None

    # 2. Loop through the number of episodes
    for episode in range(episodes):
        env.reset()  # Gym environment reset
        S = env.env.s  # 3. Getting State
        # print('First state = '+str(S))
        A = greedy_policy(Q)[S]  # 4. Deciding on first action
        # print('First action = '+str(A))
        finished = False

        cumulative_reward = 0
        # 5. Looping to the end of the episode
        while not finished:

            S_prime, reward, finished, _ = env.step(A)  # 6. Making next step
            A_prime = greedy_policy(Q)[S_prime]  # 7. Deciding on second action

            cumulative_reward = cumulative_reward + reward
            Q[S][A] = Q[S][A] + step_size * (cumulative_reward + exploration_rate * Q[S_prime][A_prime] - Q[S][A])  # 8. Update rule

            # 9. Update State and Action for the next step
            S = S_prime
            A = A_prime

            # print('state = ' + str(S))
            # print(finished)
            # print(reward)

            if(S==0):
                # reward = reward - 10
                pass

            if(S==5 or S==7 or S==11 or S==12):
                cumulative_reward = cumulative_reward - 10000
                pass
            else:
                cumulative_reward = cumulative_reward + 5

            if(S==15):
                # reward = reward + 1000
                pass

            gp = greedy_policy(Q)
        print(cumulative_reward)
        print(gp)
    return gp, Q


env = gym.make('FrozenLake-v0')

# print(env.env.desc)
# print(env.observation_space)
# print(env.action_space.n)

# one,two = sarsa(env)

# print(one)
# print(two)

# print(env.env.s)


# print(env)
# print(env.reset())

policy = utils.create_random_policy(env)
Q = create_state_action_dictionary(env, policy)

print(policy)
print(Q)

# print(list([str(y)[-2] for y in x]))
# print(str(4)[-2])

# print(lambda x: list([str(y)[-2] for y in x]), list(env.env.desc))
# print(map(lambda x: list([str(y)[-2] for y in x]), list(env.env.desc)))
# print(list(map(lambda x: list([str(y)[-2] for y in x]), list(env.env.desc))))


# print(env.env.desc[0])


# print(policy)
# print(Q)
# print(env.env.desc)
# print(env.observation_space)
# print(env.observation_space)

# print(env.action_space)
# run_game()

# def sarsa(env, episodes=100, step_size=0.01, exploration_rate=0.01):
#     policy = utils.create_random_policy(env)
#     Q = create_state_action_dictionary(env, policy)
#     for episode in range(episodes):
#         env.reset()
#         S = env.env.s
#         A = greedy_policy(Q)[S]
#         finished = False
#         while not finished:
#             S_prime, reward, finished, _ = env.step(A)
#             A_prime = greedy_policy(Q)[S_prime]
#             Q[S][A] = Q[S][A] + step_size * (reward + exploration_rate * Q[S_prime][A_prime] - Q[S][A])
#             S = S_prime
#             A = A_prime
#
#     return greedy_policy(Q), Q
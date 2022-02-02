#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9                   # Discount Factor
ALPHA = 0.2                   # Learning Rate
TEST_EPISODES = 20            # Validation Episodes


class Agent:
    def __init__(self):
        # Initialize the environment and data structures
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        # Get random action
        action = self.env.action_space.sample()
        # Obtain and return s,a,r,s'
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        # Initialize
        best_value, best_action = None, None
        # For every action
        for action in range(self.env.action_space.n):
            # Look up Q value
            action_value = self.values[(state, action)]
            # Track best action
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        # Get current state value
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        # Look up old state value
        old_v = self.values[(s, a)]
        # Blend the values
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        # Initialize
        total_reward = 0.0
        state = env.reset()
        while True:
            # Get best action
            _, action = self.best_value_and_action(state)
            # Take action
            new_state, reward, is_done, _ = env.step(action)
            # Accumulate reward
            total_reward += reward
            if is_done:
                break
            # Update current state
            state = new_state
        return total_reward


if __name__ == "__main__":
    # Initialize
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # Explore the environment
        s, a, r, next_s = agent.sample_env()
        # Update value table
        agent.value_update(s, a, r, next_s)

        # Determine average reward over a number of episodes using current policy
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        # Log data
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward

        # See if solved
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

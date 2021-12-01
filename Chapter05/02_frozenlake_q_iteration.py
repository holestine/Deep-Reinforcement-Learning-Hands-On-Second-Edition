#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        # Initialize the environment and data structures
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            # Get random action
            action = self.env.action_space.sample()
            # Take random action
            new_state, reward, is_done, _ = self.env.step(action)
            # Record the reward for the specific state, action and new state
            self.rewards[(self.state, action, new_state)] = reward
            # Increment number of times the new state was reached by taking a specific actions from a specific state
            self.transits[(self.state, action)][new_state] += 1
            # Update current state
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        # Initialize
        best_action, best_value = None, None
        # For every action
        for action in range(self.env.action_space.n):
            # Look up Q value
            action_value = self.values[(state, action)]
            # Track best action
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        # Initialize
        total_reward = 0.0
        state = env.reset()
        while True:
            # Get best actions
            action = self.select_action(state)
            # Take action
            new_state, reward, is_done, _ = env.step(action)
            # Record the reward for the specific state, action and new state
            self.rewards[(state, action, new_state)] = reward
            # Increment number of times the new state was reached by taking a specific actions from a specific state
            self.transits[(state, action)][new_state] += 1
            # Accumulate reward
            total_reward += reward
            if is_done:
                break
            # Update current state
            state = new_state
        return total_reward

    def value_iteration(self):
        # For every state
        for state in range(self.env.observation_space.n):
            # For every action
            for action in range(self.env.action_space.n):
                # Initialize action value to zero
                action_value = 0.0
                # Get counters for targets reached with the specified state and action
                target_counts = self.transits[(state, action)]
                # Get total times this state-action combo was seen
                total = sum(target_counts.values())

                for tgt_state, count in target_counts.items():
                    # Lookup the reward for the state-action-next state combination
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    # Get the best action
                    best_action = self.select_action(tgt_state)
                    # Combine current reward with discounted expected value
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    # Accumulate weighted average of action values
                    action_value += (count / total) * val
                # Store the action value for the current state and action
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    # Initialize
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")
    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        # Play randomly to populate data structures
        agent.play_n_random_steps(100)
        # Determine action values from random play
        agent.value_iteration()

        # Determine average reward over a number of episodes using current policy
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        # Log data 
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        # See if solved
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

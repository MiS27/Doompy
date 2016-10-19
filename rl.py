import random

import numpy
from lasagne.layers import get_all_param_values, set_all_param_values

from nn import create_dqn


class ReplayMemory:
    def __init__(self, capacity, input_shape, variables_size):
        states_shape = (capacity,) + input_shape
        self.size = 0
        self.index = 0
        self.capacity = capacity
        self.states_before_action = numpy.zeros(states_shape, dtype=numpy.float32)
        self.variables = numpy.zeros((capacity, variables_size), dtype=numpy.float32)
        self.actions = numpy.zeros(capacity, dtype=numpy.int32)
        self.states_after_action = numpy.zeros(states_shape, dtype=numpy.float32)
        self.variables_after_action = numpy.zeros((capacity, variables_size), dtype=numpy.float32)
        self.instant_rewards = numpy.zeros(capacity, dtype=numpy.float32)
        self.nonterminals = numpy.zeros(capacity, dtype=numpy.bool_)

    def add_experience(self, state_before_action, variables, action, state_after_action, variables_after_action,
                       instant_reward, nonterminal):
        self.states_before_action[self.index] = state_before_action
        self.variables[self.index] = variables
        self.actions[self.index] = action
        self.states_after_action[self.index] = state_after_action
        self.variables_after_action[self.index] = variables_after_action
        self.instant_rewards[self.index] = instant_reward
        self.nonterminals[self.index] = nonterminal
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity

    def get_sample(self, sample_size):
        if sample_size > self.size:
            return None
        sample = random.sample(range(0, self.size), sample_size)
        return self.states_before_action[sample], self.variables[sample], self.actions[sample], \
               self.states_after_action[sample], self.variables_after_action[sample], \
               self.instant_rewards[sample], self.nonterminals[sample]


class Agent:
    def __init__(self, n_actions, input_shape, capacity=10000, batch_size=64, start_epsilon=1.0, end_epsilon=0.1,
                 epsilon_delta=None, n_variables=0, epsilon_change_steps=2500000, target_network_update_frequency=1000):
        (network, target_network, function_learn, function_get_q_values, function_get_best_action,
         function_get_max_q_value) = create_dqn(n_actions, input_shape, n_variables)

        self.network = network
        self.target_network = target_network
        self.update_target_network()

        self.learn = function_learn
        self.get_q_values = function_get_q_values
        self.get_best_action = function_get_best_action
        self.get_max_q_value = function_get_max_q_value

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.n_variables = n_variables
        self.capacity = capacity
        self.replay_memory = ReplayMemory(capacity, input_shape, n_variables)
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_delta = self._calculate_epsilon_delta(epsilon_delta, epsilon_change_steps)

        self.target_network_update_frequency = target_network_update_frequency
        self.network_update_counter = 0

    def __getinitargs__(self):
        return self.n_actions, self.input_shape, self.capacity, self.batch_size, self.start_epsilon, \
               self.end_epsilon, self.epsilon_delta, self.n_variables

    def __getstate__(self):
        return self.epsilon

    def __setstate__(self, epsilon):
        self.epsilon = epsilon

    def load_params(self, save_file):
        params = numpy.load(save_file)
        set_all_param_values(self.network, params)
        self.update_target_network()

    def save_params(self, save_file):
        numpy.save(save_file, get_all_param_values(self.network))

    def reset_epsilon(self, epsilon_delta=None, epsilon_change_steps=None):
        self.epsilon = self.start_epsilon
        if epsilon_delta is not None or epsilon_change_steps is not None:
            self.epsilon_delta = self._calculate_epsilon_delta(epsilon_delta, epsilon_change_steps)

    def _calculate_epsilon_delta(self, epsilon_delta, epsilon_change_steps):
        if epsilon_delta is not None:
            return epsilon_delta
        else:
            return (self.start_epsilon - self.end_epsilon) / epsilon_change_steps

    def _calculate_epsilon(self):
        if self.epsilon - self.epsilon_delta > self.end_epsilon:
            self.epsilon -= self.epsilon_delta
            return self.epsilon
        else:
            return self.end_epsilon

    def choose_action(self, state, variables):
        if random.random() < self._calculate_epsilon():
            return random.randrange(0, self.n_actions)
        else:
            return self.choose_best_action(state, variables)

    def choose_best_action(self, state, variables):
        return self.get_best_action(state, variables)

    def add_experience(self, state_before_action, variables, action, state_after_action, variables_after_action,
                       reward):
        if state_after_action is not None:
            non_terminal = 1
        else:
            non_terminal = 0
            state_after_action = numpy.zeros_like(state_before_action)
            variables_after_action = numpy.zeros_like(variables)
        self.replay_memory.add_experience(state_before_action, variables, action, state_after_action,
                                          variables_after_action, reward, non_terminal)

    def learn_from_memories(self):
        sample = self.replay_memory.get_sample(self.batch_size)
        if sample:
            (states_before_action, variables, actions, states_after_action, variables_after_action, rewards,
             nonterminals) = sample
            loss = self.learn(states_before_action, variables, actions, states_after_action, variables_after_action,
                              rewards, nonterminals)
            self.network_update_counter += 1
            if self.network_update_counter % self.target_network_update_frequency == 0:
                self.update_target_network()
            return loss
        else:
            return None

    def update_target_network(self):
        set_all_param_values(self.target_network, get_all_param_values(self.network))

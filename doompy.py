import logging
from collections import deque

import collections
import json
import cv2
import numpy
import pickle
import os
import tempfile

import time

import shutil

from retrying import retry
from tqdm import trange

from rl import Agent


class CompositeState:
    def __init__(self, shape, n_variable):
        self.states = deque(maxlen=shape.frames)
        self.shape = shape
        self.variables = []
        self.n_variable = n_variable

    def __len__(self):
        return len(self.states)

    def convert_and_append(self, game_state):
        self.states.append(self._convert(game_state.image_buffer))
        self.variables = game_state.game_variables if game_state.game_variables is not None else []

    def get_data(self):
        return numpy.array(self.states).reshape([1, self.shape.frames, self.shape.y, self.shape.x]),\
               numpy.array(self.variables).astype(numpy.float32).reshape([1, self.n_variable])

    def clear(self):
        self.states.clear()

    def _convert(self, image_buffer):
        img = image_buffer[0].astype(numpy.float32) / 255.0
        img = cv2.resize(img, (self.shape.x, self.shape.y))
        return img


class Tester:
    def __init__(self, actions, input_shape,  n_variables, episodes=300, ticks=4, still_action=None, still_ticks=0):
        self.actions = actions
        self.input_shape = input_shape
        self.n_variables = n_variables
        self.episodes = episodes
        self.ticks = ticks
        self.still_action = still_action
        self.still_ticks = still_ticks

    def _test_episode(self, game, agent):
        game.new_episode()
        composite_state = CompositeState(self.input_shape, self.n_variables)
        counter = 0
        actions = []
        rewards = []
        while not game.is_episode_finished():
            # time.sleep(1)
            counter += 1
            if hasattr(self, 'debug'):
                print 'state:', game.get_state().image_buffer
            composite_state.convert_and_append(game.get_state())
            if len(composite_state) == self.input_shape.frames:
                state_before_action, variables = composite_state.get_data()
                if hasattr(self, 'debug'):
                    print 'composite_state:', state_before_action
                action = agent.choose_best_action(state_before_action, variables)
                if hasattr(self, 'debug'):
                    print 'action:', action
                rewards.append(game.make_action(self.actions[action], self.ticks))
                if self.still_action:
                    rewards.append(game.make_action(self.still_action, self.still_ticks))
                actions.append(action)
            else:
                game.advance_action(self.ticks + self.still_ticks)
                rewards.append(game.get_last_reward())
        #if game.get_total_reward() < -100:
        logging.info("Counter %f, Reward %f", counter, game.get_total_reward())
        # logging.info("Actions: %s", ", ".join(str(x) for x in actions))
        # logging.info("Rewards: %s", ", ".join(str(x) for x in rewards))
        return game.get_total_reward()

    #@retry(stop_max_attempt_number=10, wait_random_min=1000, wait_random_max=2000)
    def test(self, game, agent, complexity=None):
        try:
            game.init()
            if complexity is not None:
                game.send_game_command("set complexity " + str(complexity))
            else:
                complexity = -1
            rewards = []
            for j in trange(self.episodes, desc='test episodes'):
                reward = self._test_episode(game, agent)
                # print reward
                rewards.append(reward)
        finally:
            game.close()
        test_rewards = numpy.array(rewards)
        logging.info("TEST: complexity: %d, results: mean: %f std: %f max: %f min: %f", complexity,
                     test_rewards.mean(), test_rewards.std(), test_rewards.max(), test_rewards.min())
        return test_rewards


class Tutor:
    def __init__(self, agent, actions, input_shape, n_variables, ticks=4, still_action=None, still_ticks=0):
        self.actions = actions
        self.ticks = ticks
        self.shape = input_shape
        self.n_variables = n_variables
        self.agent = agent
        self.still_action = still_action
        self.still_ticks = still_ticks

    #@retry(stop_max_attempt_number=10, wait_random_min=1000, wait_random_max=2000)
    def epoch(self, game, steps, complexity=None):
        try:
            game.init()
            if complexity is not None:
                game.send_game_command("set complexity " + str(complexity))
            epoch_start = time.time()
            game.new_episode()
            full_episodes = 0
            composite_state = CompositeState(self.shape, self.n_variables)
            for j in trange(steps, desc='steps'):
                while len(composite_state) < self.shape.frames:
                    if game.is_episode_finished():
                        game.new_episode()
                        composite_state.clear()
                    else:
                        composite_state.convert_and_append(game.get_state())
                        game.advance_action(self.ticks + self.still_ticks)

                state_before_action, variables = composite_state.get_data()
                action, reward = self._perform_action(game, state_before_action, variables)
                if game.is_episode_finished():
                    state_after_action = None
                    variables_after_action = None
                    game.new_episode()
                    composite_state.clear()
                    full_episodes += 1
                else:
                    composite_state.convert_and_append(game.get_state())
                    state_after_action, variables_after_action = composite_state.get_data()
                self.agent.add_experience(state_before_action, variables, action, state_after_action,
                                          variables_after_action, reward * 0.01)
                self.agent.learn_from_memories()

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            logging.info("full episodes played: %d", full_episodes)
            logging.info("epoch time: %f", epoch_time)
            logging.info("epoch final epsilon: %f", self.agent.epsilon)
            self.agent.update_target_network()
        finally:
            game.close()

    def _perform_action(self, game, state_before_action, variables):
        action = self.agent.choose_action(state_before_action, variables)
        reward = game.make_action(self.actions[action], self.ticks)
        if self.still_action:
            reward += game.make_action(self.still_action, self.still_ticks)
        return action, reward


class Runner:
    def __init__(self, run_name, state_file, actions, input_shape, n_variables, start_epsilon, complexities, learning_steps_per_epoch,
                 testing_episodes, max_complexity, ticks, still_action, still_ticks, log_file, load_net_file,
                 save_net_file, save_net_file_format):
        self.actions = actions
        self.state_file = state_file
        self.run_name = run_name
        self.log_file = log_file or run_name + ".log"
        self.load_net_file = load_net_file or run_name + ".npy"
        self.save_net_file = save_net_file
        self.save_net_file_format = save_net_file or save_net_file_format

        self.agent = Agent(len(self.actions), input_shape, n_variables=n_variables, start_epsilon=start_epsilon)

        self.tester = Tester(self.actions, input_shape, n_variables, ticks=ticks, still_action=still_action,
                             still_ticks=still_ticks, episodes=testing_episodes)
        self.tutor = Tutor(self.agent, self.actions, input_shape, n_variables, ticks=ticks, still_action=still_action,
                           still_ticks=still_ticks)

        self.epochs_done = 0
        self.complexities = complexities
        self.max_complexity = max_complexity
        self.complexity_done = -1
        self.learning_steps_per_epoch = learning_steps_per_epoch

    #@retry(stop_max_attempt_number=10, wait_random_min=1000, wait_random_max=10000)
    def run(self, game, esp_factor=2):
        logging.basicConfig(filename=self.log_file, level=logging.INFO, filemode='a')
        if os.path.exists(self.load_net_file):
            logging.info("Loading net file: %s", self.load_net_file)
            self.agent.load_params(self.load_net_file)
        for complexity, epochs in [(x, y) for x, y in self.complexities if x > self.complexity_done]:
            logging.info('Complexity: %d', complexity)
            if self.epochs_done == 0:
                self.agent.reset_epsilon(epsilon_change_steps=epochs * self.learning_steps_per_epoch / esp_factor)
            if self.save_net_file_format:
                self.save_net_file = self.save_net_file_format.format(self.run_name, complexity)
            for i in trange(self.epochs_done + 1, epochs + 1, desc='epochs'):
                logging.info("epoch: %d", i)
                self.tutor.epoch(game, self.learning_steps_per_epoch, complexity)
                self.tester.test(game, self.agent, complexity)
                if self.max_complexity is not None and complexity != self.max_complexity:
                    self.tester.test(game, self.agent, self.max_complexity)
                self.epochs_done = i
                self._persist()
            self.epochs_done = 0
            self.complexity_done = complexity

    def _persist(self):
        self.load_net_file = self.save_net_file
        tmp = tempfile.mkdtemp(dir='.') + '/'
        self.agent.save_params(tmp + self.save_net_file)
        pickle.dump(self, open(tmp + self.state_file, 'wb'))
        os.rename(tmp + self.save_net_file, self.save_net_file + '.saved')
        os.rename(tmp + self.state_file, self.state_file + '.saved')
        shutil.rmtree(tmp)
        os.rename(self.save_net_file + '.saved', self.save_net_file)
        os.rename(self.state_file + '.saved', self.state_file)

    @staticmethod
    def prepare_runner(run_name, state_file, actions, input_shape, complexities, n_variables=0, start_epsilon=1.0,
                       learning_steps_per_epoch=50000, testing_episodes=300, max_complexity=None, ticks=4,
                       still_action=None, still_ticks=0, log_file=None, load_net_file=None, save_net_file=None,
                       save_net_file_format="{}_{}.npy"):
        state_file_saved = state_file + '.saved'
        if os.path.exists(state_file_saved):
            os.rename(state_file_saved, state_file)
            runner = pickle.load(open(state_file, 'rb'))
            if os.path.exists(runner.load_net_file + '.saved'):
                os.rename(runner.load_net_file + '.saved', runner.load_net_file)
        elif os.path.exists(state_file):
            runner = pickle.load(open(state_file, 'rb'))
        else:
            runner = Runner(run_name, state_file, actions, input_shape, n_variables, start_epsilon, complexities,
                            learning_steps_per_epoch, testing_episodes, max_complexity, ticks, still_action, still_ticks, log_file,
                            load_net_file, save_net_file, save_net_file_format)
        return runner


def update_config(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update_config(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def load_config(filename, config=None):
    if config is None:
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return update_config(config, load_config(filename))

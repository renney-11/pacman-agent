# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  This is an improved offensive agent that seeks food, returns home when carrying enough food,
  and avoids ghosts.
  """
  
    FOOD_RETURN_THRESHOLD = 4  # Number of food to carry before returning home

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance # Encourage getting closer to food

        # Number of carried food
        carried_food = my_state.num_carrying
        features['carried_food'] = carried_food # Encourage carrying food

        # If pacman is carrying enough food, we encourage it to return home
        if carried_food >= self.FOOD_RETURN_THRESHOLD:
            # Distance to home
            home_dist = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = home_dist # Encourage returning home
        else:
            # When not carrying enough food, do not consider distance to home
            features['distance_to_home'] = 0

        # Avoid ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        # Distance to nearest ghost
        if len(ghosts) > 0:
            # Calculate distances to all ghosts
            ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_dist = min(ghost_distances)
            features['nearest_ghost'] = min_ghost_dist # Prefer being far from ghosts
            # Penalize being too close to a ghost
            if min_ghost_dist <= 4:
                features['ghost_threat'] = 1 # Strong penalty for being close to a ghost
            else:
                features['ghost_threat'] = 0
        else:
            features['nearest_ghost'] = 10 # if no ghosts, set a default safe distance
            features['ghost_threat'] = 0 # No threat if no ghosts

        # Discourage stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1 # Penalize stopping
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1 # Penalize reversing

        # Encourage returning home when time is low
        time_left = game_state.data.timeleft if hasattr(game_state.data, 'timeleft') else None
        if time_left is not None and time_left < 100 and my_state.num_carrying > 0:
            home_dist = self.get_maze_distance(my_pos, self.start)
            features['low_time_home'] = home_dist
        else:
            features['low_time_home'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100, # Strongly encourage eating food
            'distance_to_food': -1, # Encourage getting closer to food
            'carried_food': 10, # Encourage carrying food
            'distance_to_home': -2, # Encourage returning home when carrying food
            'nearest_ghost': 2, # Prefer being far from ghosts
            'ghost_threat': -1000, # Strong penalty for being close to a ghost
            'stop': -100, # Discourage stopping
            'reverse': -2, # Discourage reversing
            'low_time_home': -5,  # Encourage returning home when time is low
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    An improved defensive agent that chases invaders, patrols food, and avoids crossing into enemy territory.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # Patrol around your food when no invaders are visible
            food_list = self.get_food_you_are_defending(successor).as_list()
            # Distance to nearest food
            if len(food_list) > 0:
                min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['food_distance'] = min_food_dist # Encourage patrolling food
            else:
                features['food_distance'] = 0 # No food to defend

        # Discourage stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000, # Strong penalty for invaders
            'on_defense': 100, # Reward for staying on defense
            'invader_distance': -10, # Chase invaders
            'food_distance': -1, # Patrol food when no invaders
            'stop': -100, # Discourage stopping
            'reverse': -2 # Discourage reversing
        }
        

import copy
import heapq
import math
import numpy as np

class TreeNode:
    def __init__(self, parent, game_state, log_pi, d, action):
        self._game_state = game_state
        self._log_pi = log_pi
        self._d = d
        self._levin_cost = 0
        self._action = action
        self._parent = parent
        self._probabilitiy_distribution_a = None
    
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the 
        game state in the nodes. 
        """
        return self._game_state == other._game_state
    
    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        return self._levin_cost < other._levin_cost    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def set_probability_distribution_actions(self, d):
        self._probabilitiy_distribution_a = d
        
    def get_probability_distribution_actions(self):
        return self._probabilitiy_distribution_a
    
    def set_levin_cost(self, c):
        self._levin_cost = c
    
    def get_log_pi(self):
        """
        Returns the pi cost of a node
        """
        return self._log_pi
    
    def get_depth(self):
        """
        Returns the pi cost of a node
        """
        return self._d
    
    def get_game_state(self):
        """
        Returns the game state represented by the node
        """
        return self._game_state
    
    def get_parent(self):
        """
        Returns the parent of the node
        """
        return self._parent
    
    def get_action(self):
        """
        Returns the action taken to reach node stored in the node
        """
        return self._action

class Trajectory():
    def __init__(self, states, actions):
        self._states = states
        self._actions = actions
        
    def get_states(self):
        return self._states
    
    def get_actions(self):
        return self._actions
    
    def length(self):
        return len(self._states)
    
class LevinTS():
    def __init__(self):
        self.OPEN = []
        self.CLOSED = {}
            
    def recover_path(self, tree_node):
        states = []
        actions = []
        
        state = tree_node.get_parent()
        action = tree_node.get_action()
        
        while not state.get_parent() is None:
            states.append(state.get_game_state())
            actions.append(action)
            
            action = state.get_action()
            state = state.get_parent()
            
        states.append(state.get_game_state())
        actions.append(action)
        
        return Trajectory(states, actions)  

    def get_levin_cost(self, node):
        return np.log(node.get_depth()) - node.get_log_pi()

    def search(self, state, model, budget):      
        self.OPEN = []
        self.CLOSED = {}
        root = TreeNode(None, state, 0, 0, -1)
        heapq.heappush(self.OPEN, root)

        while self.OPEN != []:
            parent_node = heapq.heappop(self.OPEN)
            if parent_node.__hash__() in self.CLOSED:
                continue
            self.CLOSED[parent_node.__hash__()] = parent_node

            parent = parent_node.get_game_state()
            if parent.is_solution():
                return parent_node._levin_cost, len(self.CLOSED), self.recover_path(parent_node)

            if len(self.CLOSED) >= budget:          # budget = expansions
                return -1, len(self.CLOSED), None

            if parent.has_tip_reached_goal():            # dead end: at goal but not a solution
                continue                                 # reduce computational cost

            # policy is conditioned on the PARENT state's pattern
            log_probs = np.log(model.get_probabilities(parent.get_pattern()))

            for a in parent.successors_parent_pruning(parent_node.get_action()):
                child = copy.deepcopy(parent)
                child.apply_action(a)
                child_node = TreeNode(parent_node, child,            # node, not s
                                      parent_node.get_log_pi() + log_probs[a],
                                      parent_node.get_depth() + 1, a)
                if child_node.__hash__() not in self.CLOSED:
                    child_node.set_levin_cost(self.get_levin_cost(child_node))
                    heapq.heappush(self.OPEN, child_node)

        return -1, len(self.CLOSED), None
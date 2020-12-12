# adversarialAgents.py
# --------------
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
#
# Modified for use at University of Bath.


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X
        for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                       if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class AdversarialSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    adversarial searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent and AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """




    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

#We have inherited the scoreEvaluationFunction and the depth.

def GameTerminated(current_game_state):
    return current_game_state.isWin() or current_game_state.isLose()
def ActionsExhausted(actions):
    return len(actions) == 0
PACMAN_INDEX = 0
START_DEPTH = 0
class MinimaxAgent(AdversarialSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def Search(self, current_game_state, current_depth, maximising_):
        node_value = self.EvaluationFunction()
        #We need to get the value of the current game state by querying the evaluation function
        if current_depth == 0:
            return node_value
    
    def MaximisingAgent(self, current_game_state, current_depth):
        #Explanation : Obtain the valid actions that pacman can take.
        pacman_valid_actions = current_game_state.getLegalActions(0)
        #Explanation : Pacman is trying to MAXIMISE his score. We want to keep track of the action which gets us the maximum score.
        best_score = float('-inf')
        best_action = None
        #Explanation : Recursion termination if the game has finished, or pacman has no further actions, or if we have reached depth. Leaf node.
        if GameTerminated(current_game_state) or len(pacman_valid_actions) == 0 or current_depth == self.depth:
            return self.evaluationFunction(current_game_state)
        #Explanation : Iterate through each action in search for the best one.
        for current_action in pacman_valid_actions:
            #Explanation : After pacman's turn, it's always the ghost's turn next.
            action_score = self.MinimisingAgent(current_game_state.generateSuccessor(0,current_action), current_depth, PACMAN_INDEX + 1)
            #Explanation : We want the MAX score.
            if action_score < best_score:
                continue
            best_score = action_score
            best_action = current_action
        #Explanation : If the depth is not the start, that means we still need to create subtrees. Return the score.
        return best_action if current_depth == START_DEPTH else best_score

    def MinimisingAgent(self, current_game_state, current_depth, current_agent_index):
        #Explanation : Obtrain the valid actions for this ghost.
        ghost_valid_actions = current_game_state.getLegalActions(current_agent_index)
        #Explanation : Recursion termination when game is won/lost or when the ghost has no moves to make. Leaf node.
        if len(ghost_valid_actions) == 0 or GameTerminated(current_game_state):
            return self.evaluationFunction(current_game_state)
        #Explanation : The ghosts want to MINIMISE the score. We want to get the action which constitutes for the minimum score.
        best_score = float('inf')
        best_action = None
        #Explanation : Get the number of agents, since we will need to know when to get back into a max layer.
        n_agents = current_game_state.getNumAgents()
        #Explanation : Iterate through each action in search for the action which gives the lowest score.
        for current_ghost_action in ghost_valid_actions:
            #Explanation : Obtain the future game state after this action.
            future_game_state = current_game_state.generateSuccessor(current_agent_index, current_ghost_action)
            #Explanation : If we are not the last ghost, then it will be another ghost's turn after this.
            #Explanation : If we are the last ghost, then it will be pacmans turn next.
            action_value = self.MinimisingAgent(future_game_state, current_depth, current_agent_index + 1) if current_agent_index < n_agents - 1 else self.MaximisingAgent(future_game_state, current_depth + 1)
            #Explanation : We want the MIN score.
            if action_value > best_score:
                continue
            best_action = current_ghost_action
            best_score = action_value
        #Explanation : Note that we never return an action from MinimisingAgent, since we are in the middle of the tree!
        return best_score
    
    def getAction(self, state):
        #Explanation : Initialise the search by querying the max value of the start node.
        return self.MaximisingAgent(state,START_DEPTH)






def scoreEvaluation(state):
    return state.getScore()
class AlphaBetaAgent(AdversarialSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, current_game_state):
        #Explanation : Obtain pacman's valid actions.
        pacman_legal_actions = current_game_state.getLegalActions(0)
        #Explanation : We want the action which gives the highest alpha. 
        best_action = None
        action_alpha = float('-inf')
        ##Explanation : Alpha is the minimum value of the maximising player, ie the lower bound of a given parent node.
        current_alpha = float('-inf')
        ##Explanation : Beta is the maximum value of the minimising player, ie the upper bound of a given parent node.
        current_beta = float('inf')

        #Explanation : Iterate through of the valid actions of pacman in search for the best action
        for current_action in pacman_legal_actions:
            #Explanation : Compute the future game state with this action
            future_game_state = current_game_state.generateSuccessor(PACMAN_INDEX, current_action)
            #Explanation : Ghosts always come after pacman.
            action_alpha = self.Minimising_Agent(current_alpha, current_beta, future_game_state,0, 1)
            #Explanation : Definition of alpha.
            #Amendment made for strict inqualities
            if current_alpha < action_alpha:
                ##Explanation : If so, set the record alpha to the obtained one, and log the action which produced the result.
                current_alpha = action_alpha
                best_action = current_action
        #Explanation : Return the best action which constitutes for the highest alpha.
        return best_action
    def Minimising_Agent(self, current_alpha, current_beta, current_game_state,current_depth,current_player_index):
        #Explanation : Obtain the legal actions of the ghost.
        ghost_valid_actions = current_game_state.getLegalActions(current_player_index)
        #Explanation : If we have no more actions left, or the game has terminated, then we are simply a leaf node. Return the result of the evaluation function.
        #Note that a depth check isn't necessary here since the depth is only increased on pacmans turn.
        if ActionsExhausted(ghost_valid_actions) or GameTerminated(current_game_state):
            return self.evaluationFunction(current_game_state)
        #Explanation : Keep track of the lowest beta.
        action_beta = float('inf')
        #Explanation : We need to compare the current agent index to this number to determine when its going to be pacmans turn.
        NUM_AGENTS = current_game_state.getNumAgents()
        #Explanation : Iterate through each action in search for a best action.
        for current_action in ghost_valid_actions:
            #Explanation : Generate the future game state.
            future_game_state = current_game_state.generateSuccessor(current_player_index, current_action)
            #Explanation : Since we can have an arbritrary number of ghosts, we need to process all but the last ghost as min layers.
            #Explanation : After the last ghost's turn, it will be the max agents turn, in which case the depth is increased (a full cycle is reached)
            action_beta = min(action_beta, self.Minimising_Agent(current_alpha, current_beta, future_game_state, current_depth, current_player_index + 1) if current_player_index < NUM_AGENTS - 1 else self.Maximising_Agent(current_alpha, current_beta, future_game_state, current_depth + 1))
            #Explanation : Record alpha if best.
            #Amendment made for strict inqualities
            if action_beta < current_alpha:
                return action_beta
            #Explanation : Update the propagation value of beta.
            current_beta = min(current_beta, action_beta)
        return action_beta
    def Maximising_Agent(self, current_alpha, current_beta, current_game_state, current_depth):
        #Explanation : Obtain the valid actions that pacman is able to take.
        pacman_valid_actions = current_game_state.getLegalActions(PACMAN_INDEX)
        #Explanation : Keep track of the record alpha.
        action_alpha = float('-inf')
        #Explanation : Recursive termination when pacman can't make any moves, or we have reached depth, or if the game has finished. Leaf node.
        if ActionsExhausted(pacman_valid_actions) or current_depth == self.depth or GameTerminated(current_game_state):
            return self.evaluationFunction(current_game_state)
        #Explanation : Iterate through each action in the search for the best one.
        for current_action in pacman_valid_actions:
            #Explanation : Compute the future game state with the current action.
            future_game_state= current_game_state.generateSuccessor(PACMAN_INDEX, current_action)
            #Explanation : By definition, alpha has to be the MAXIMUM value of the minimising agent. (Minimising agents turn next.)
            action_alpha = max(action_alpha,self.Minimising_Agent(current_alpha, current_beta, future_game_state, current_depth,1))
            #Explanation : Record alpha
            #Amendment made for strict inqualities
            if action_alpha > current_beta:
                return action_alpha
            #Explanation : Update propagation value of alpha.
            current_alpha = max(current_alpha, action_alpha)
        return action_alpha
def betterEvaluationFunction(current_game_state):
    #There are two types of objects in pacman.
        #Things which are good for pacman.
        #Things which are detrimental for pacman.
    #If pacman gets closer to a detrimental object, the score should decrease.
    #If pacman gets closer to a good object, then the score should increase.
    
    #This is the model by which we have based the evaluation function.


    #Explanation : If pacman is doing bad, then the evaluation will decrease.
    game_score = current_game_state.getScore()
    #Explanation : Obtain the position of the pacman, since we need to compute the distance between him and some other things in the map.
    pacman_position = current_game_state.getPacmanPosition()


    #Explanation : Euclidean Distance between two tuples (x,y), (z,k)
    def EuclideanDistance(pos1, pos2):
       return math.sqrt(math.pow(int(pos1[0])-int(pos2[0]), 2) + math.pow(int(pos1[1]) - int(pos2[1]),2))

    #Explanation : Accepts an array of position tuples.
    #Explanation : The radius of vision can be specified.
    def ClosestEntity(entity_position_array, radius):
       #Explanation : Store the distances from the pacman to the entities.
       entity_distances = []
       #Explanation : Initialise the distance array.
       for entity_position in entity_position_array:
           #Explanation : Entity is too far.
           if EuclideanDistance(entity_position, pacman_position) > radius:
               continue
            #Explanation : Entity in range.
           entity_distances.append(entity_position)
        #Explanation : All entities are out of range
       if len(entity_distances) == 0:
           return (False, 0)
        #Explanation : Non empty return with possibly non zero distance. 
       return (True, EuclideanDistance(min(entity_distances, key = lambda pos : EuclideanDistance(pos, pacman_position)), pacman_position))

    #Explanation : The entity metrics by which we will compute the closest of each.
    game_score = current_game_state.getScore()
    food_array = current_game_state.getFood().asList()
    capsule_array = current_game_state.getCapsules()
    ghost_positions = current_game_state.getGhostPositions()
    scared_ghost_positions = [scared_ghost_pos.getPosition() for scared_ghost_pos in current_game_state.getGhostStates() if scared_ghost_pos.scaredTimer > 0]
    
    closest_food = ClosestEntity(food_array, float('inf'))
    closest_ghost = ClosestEntity(ghost_positions,2)
    closest_capsule = ClosestEntity(capsule_array, float(1.42))
    closest_scared_ghost = ClosestEntity(scared_ghost_positions, float('inf'))
    final_score = game_score
    remaining_capsules = (True,len(capsule_array))
    remaining_food = (True,len(food_array))

    #Explanation : A function which models the "the closer the entity, the higher the score" behaviour.
    def CloserBetter(entity):
        if entity[0] is False:
            #The food is out of range.
            return 0
        else:
            if entity[1] == 0:
                return 1
        return 1/entity[1]
    #Explanation : A function which models the "the farther the entity, the better the score" behaviour.
    def FartherBetter(entity):
        if entity[0] is False:
            return 1
        else:
            if entity[1] == 0:
                return -1
        return (-1/(entity[1]))+1
    #Explanation : Result is a combination of all of the above parameters.
    return game_score + 3*CloserBetter(closest_food) + 120*CloserBetter(closest_scared_ghost) + 0.01*FartherBetter(closest_ghost) + 20*CloserBetter(closest_capsule) + 0.1*CloserBetter(remaining_food) + -50*remaining_capsules[1]




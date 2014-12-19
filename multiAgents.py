# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 

# educational purposes provided that (1) you do not distribute or publish 

# solutions, (2) you retain this notice, and (3) you provide clear 

# attribution to UC Berkeley, including a link to 

# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

# 

# Attribution Information: The Pacman AI projects were developed at UC Berkeley.

# The core projects and autograders were primarily created by John DeNero 

# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

# Student side autograding was added by Brad Miller, Nick Hay, and 

# Pieter Abbeel (pabbeel@cs.berkeley.edu).



from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
		
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print"successors=",successorGameState
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print "current state=",currentGameState
        print "successors=",successorGameState
        print " scaredtimes=",newScaredTimes
        print " ghoststates=",newGhostStates
        print " newpos= ",newPos
        print " newfood= ",newFood

        
        "*** YOUR CODE HERE ***"""
	
	#USeful information extracted from the Game state
	successorGameState = currentGameState.generatePacmanSuccessor(action)
	newPos = successorGameState.getPacmanPosition()
	newFood = successorGameState.getFood()
	newGhostStates = successorGameState.getGhostStates()
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

	#Minimum distance from ghost
	minGhostDist = 'Inf'
	maxGhostDist = 0
	for ghostState in newGhostStates:
		minGhostDist = min(minGhostDist, manhattanDistance(newPos, ghostState.getPosition()))
		maxGhostDist = max(maxGhostDist, manhattanDistance(newPos, ghostState.getPosition()))
	
	#Minimum distance from Food
	minFoodDist = 'Inf'
	foodSum = 0
	newFoodList = newFood.asList()	
	for foodPos in newFoodList:
		minFoodDist = min(minFoodDist, manhattanDistance(newPos, foodPos))
		foodSum = foodSum + manhattanDistance(newPos, foodPos)


	if minFoodDist == 0:
		scoreValue = 0
	else:

		# Give high score to the poistion with minimum distance from the food by taking the reciprocal of minFoodDist
		scoreValue = 1.0/(500.0*float(minFoodDist))
		# Discard all moves with distance from nearest ghost less than 2 (by giving high negative value) so that pacman does not die
		if minGhostDist < 2:
			scoreValue = -100000

	return scoreValue + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """
    def successors(self, gameState, agentInd):
        """
        Returns a list of game states one action away for agentInd.
        """
        actions = gameState.getLegalActions(agentInd)
        return [(gameState.generateSuccessor(agentInd, action), action) for action in actions]
        
        
    def terminalTest(self, state, depth):
        """
        Returns true if state is a terminal state or max depth is reached.
        """
        if state.isWin() or state.isLose() or self.depth == depth:
            return True
        return False
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
       
	#Computing max score given the depth, Game state and number of ghosts  
    	def maxvalue(gameState, depth, numghosts):
        	if gameState.isWin() or gameState.isLose() or depth == 0:
        		return self.evaluationFunction(gameState)
        	v = -(float("inf"))
		#Checking for all legal actions for the pacman
        	legalActions = gameState.getLegalActions(0)
		#Generating successor moves for every legal action
        	for action in legalActions:
        		if action != Directions.STOP:
        			v = max(v, minvalue(gameState.generateSuccessor(0, action), depth, 1, numghosts))
		#Returning the maximum score
        	return v
    
	#Computing min score given the depth, Game state and number of ghosts  
    	def minvalue(gameState, depth, agentindex, numghosts):
        	if gameState.isWin() or gameState.isLose() or depth == 0:
        		return self.evaluationFunction(gameState)
        	v = float("inf")
		#Checking for all the legal actions for current ghost
        	legalActions = gameState.getLegalActions(agentindex)
		#Calling the maxvalue function if the current ghost is the last ghost else calling the minvalue function
        	if agentindex == numghosts:
        		for action in legalActions:
        			if action != Directions.STOP:
        				v = min(v, maxvalue(gameState.generateSuccessor(agentindex, action), depth - 1, numghosts))
        	else:
        		for action in legalActions:
        			if action != Directions.STOP:
        				v = min(v, minvalue(gameState.generateSuccessor(agentindex, action), depth, agentindex + 1, numghosts))
		#Returning the minimum score
        	return v
    	
	#Getting a list of all legal actions
	legalActions = gameState.getLegalActions()
	
	#Computing the total number of ghosts
    	numghosts = gameState.getNumAgents() - 1
    	bestaction = Directions.STOP
    	score = -(float("inf"))
    	for action in legalActions:
		#Removing the Directions.STOP action from Pac-Man's list of possible actions
        	if action!=Directions.STOP:
        		nextState = gameState.generateSuccessor(0, action)
        		prevscore = score
			#Getting the scores from the successor game states
        		score = max(score, minvalue(nextState, self.depth, 1, numghosts))
        		if score > prevscore:
        			bestaction = action
    	return bestaction

class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
	
	#Computing maximum score
	def maxvalue(gameState, alpha, beta, depth):
        	if gameState.isWin() or gameState.isLose() or depth == 0:
            		return self.evaluationFunction(gameState)
		#Initial Lower bound for alpha
        	alpha_tmp = -(float("inf"))
		#Check for all legal actions of pacman
        	legalActions = gameState.getLegalActions(0)
        	for action in legalActions:
			if action != Directions.STOP:
				nextState = gameState.generateSuccessor(0, action)
				alpha_tmp = max(alpha_tmp, minvalue(nextState, alpha, beta, gameState.getNumAgents() - 1, depth))
				#Pruning step
				if  alpha_tmp >= beta:
					return alpha_tmp
				#Choose the maximum value from alpha and the returned value from successor
				alpha = max(alpha, alpha_tmp)
		return alpha_tmp
   
	#Computing min score 
	def minvalue(gameState, alpha, beta, agentindex, depth):
		#Compute the number of ghosts
		numghosts = gameState.getNumAgents() - 1
		if gameState.isWin() or gameState.isLose() or depth == 0:
			return self.evaluationFunction(gameState)
		#Initial Upper bound for beta
		beta_tmp = float("inf")
		#Check for all legal actions of ghost
		legalActions = gameState.getLegalActions(agentindex)
		for action in legalActions:
			if action != Directions.STOP:
				nextState = gameState.generateSuccessor(agentindex, action)
				#Call the maxvalue function if the current ghost is the last ghost else calling the minvalue function
				if agentindex == numghosts:
					beta_tmp = min(beta_tmp, maxvalue(nextState, alpha, beta, depth - 1))
					#Pruning step
					if beta_tmp <= alpha:
						return beta_tmp
					#Choose the minimum value from beta and the returned value from successor
					beta = min(beta, beta_tmp)
				else:
					beta_tmp = min(beta_tmp, minvalue(nextState, alpha, beta, agentindex + 1, depth))
					#Pruning step
					if beta_tmp <= alpha:
						return beta_tmp
					#Choose the minimum value from beta and the returned value from successor
					beta = min(beta, beta_tmp)
		return beta_tmp
    
    

	#Compute all the legal actions
	legalActions = gameState.getLegalActions(0)
	bestaction = Directions.STOP
	#initial score (same as alpha value)
	score = -(float("inf"))
	alpha = -(float("inf"))
	beta = float("inf")
	for action in legalActions:
		#Removing the Directions.STOP action from Pac-Man's list of possible actions
		if action!=Directions.STOP:
			nextState = gameState.generateSuccessor(0, action)
			prevscore = score
			#Getting the scores of the successor game states
			score = max(score, minvalue(nextState, alpha, beta, 1, self.depth))
			if score > prevscore:
				bestaction = action
			if score >= beta:
				return bestaction
			alpha = max(alpha, score)
	return bestaction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


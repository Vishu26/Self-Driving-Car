import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor


        self.count = 1


    def reset(self, destination=None, testing=False):

        import numpy as np
        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        if testing ==True :
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
        	self.epsilon = math.cos(0.01*self.count)
	        self.count+=1
        return None

    def build_state(self):

        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline


    
        state = (waypoint,inputs['light'], inputs['left'], inputs['right'], inputs['oncoming'])
        return state


    def get_maxQ(self, state):


        maxQ = -float('inf')

        for act in self.Q[state]:
            if self.Q[state][act]>maxQ:
                maxQ = self.Q[state][act]


        return maxQ 


    def createQ(self, state):

        if self.learning and (state not in self.Q):
            self.Q[state] = {None : 0.0, 'left' : 0.0, 'right' : 0.0, 'forward' : 0.0}
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """
        import random
        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        
        action = []

        


        if self.learning == False:
            action.append(random.choice(self.valid_actions))
        else:
            rand = random.random()
            if rand <=self.epsilon:
                action.append(random.choice(self.valid_actions))
            else :
                max_action = self.get_maxQ(state)

                for act in self.Q[state]:
                    if self.Q[state][act]==max_action:
                        action.append(act)
        return random.choice(action)


    def learn(self, state, action, reward):

        if self.learning == True:
            self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * reward
        return


    def update(self):


        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        print(self.Q)
        return
        

def run():

    env = Environment()
    

    agent = env.create_agent(LearningAgent, learning=True)

    env.set_primary_agent(agent, enforce_deadline=True)


    sim = Simulator(env, update_delay=0.01, log_metrics=True, optimized=True, display=True)
    

    sim.run(n_test=10, tolerance=0.005)


if __name__ == '__main__':
    run()

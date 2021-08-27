import numpy
import gym

class BitFlip:
 
    def __init__(self, size = 30):
        self.size           = size

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(self.size, ))
        self.goal_space 		    = gym.spaces.Box(low=-1, high=1, shape=(self.size, ))
        self.action_space 	        = gym.spaces.Discrete(self.size)

        self.reset()

    def reset(self):
        self.goal      = numpy.random.randint(0, 2, (self.size))
        self.state     = numpy.random.randint(0, 2, (self.size))
            
        self.steps      = 0
        
        return self._update_observation()
            
    def step(self, action):
        reward = 0
        done   = False
        info   = None

        self.steps+= 1

        actions         = numpy.zeros((self.size), dtype=int)
        actions[action] = 1 

        self.state = (self.state + actions)%2

        dif = ((self.state - self.goal)**2).sum()

        if dif < 0.01:
            done    = True
            reward  = 1.0
        elif self.steps > 8*self.size:
            done       = True

        return self._update_observation(), reward, done, info

    def render(self):     
        print("state  = ", self.state)  
        print("goal   = ", self.goal)  
        print("\n\n")
        
    def _update_observation(self):
        result = {}
        result["observation"]   = self.state.copy().astype(numpy.float32)
        result["achieved_goal"] = self.state.copy().astype(numpy.float32)
        result["desired_goal"]  = self.goal.copy().astype(numpy.float32)
   
        return result

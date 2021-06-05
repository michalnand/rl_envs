import numpy
import gym
import cv2

class TunnelEnv:
 
    def __init__(self, envs_count = 16, size = 128):
        self.envs_count     = envs_count 
        self.size           = size
        self.height         = 7

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(2, ))
        self.action_space 	        = gym.spaces.Discrete(5)

        self.tunnel = numpy.zeros((self.height, self.size))
 
        for i in range(size):
            self.tunnel[0][i]              = -1.0
            self.tunnel[self.height-1][i]  = -1.0

            if i > 0:
                if i%32 == 16:
                    self.tunnel[1][i] = -1.0
                    self.tunnel[2][i] = -1.0
                    self.tunnel[3][i] = -1.0
                elif i%32 == 0:
                    self.tunnel[3][i] = -1.0
                    self.tunnel[4][i] = -1.0
                    self.tunnel[5][i] = -1.0

        self.tunnel[self.height//2][size-1] = 1.0


        self.reset()

    def reset(self, env_id = -1):
        if env_id == -1:
            self.positions_x = numpy.zeros((self.envs_count), dtype=int)
            self.positions_y = numpy.zeros((self.envs_count), dtype=int)

            for e in range(self.envs_count):
                self.positions_x[e] = 0
                self.positions_y[e] = self.height//2

            self.steps     = numpy.zeros(self.envs_count, dtype=int)
            
        else:
            self.positions_x[env_id]    = 0
            self.positions_y[env_id]    = self.height//2
            self.steps[env_id]          = 0

        if env_id != -1:
            obs = self._update_observations()[env_id]
        else:
            obs = self._update_observations()
            
        return obs


    def step(self, actions):
        rewards = numpy.zeros(self.envs_count, dtype=numpy.float32)
        dones   = numpy.zeros(self.envs_count, dtype=bool)
        infos   = numpy.zeros(self.envs_count, dtype=numpy.float32)

        self.steps+= 1

        actions    = numpy.array(actions)

        self.positions_x+=  1*(actions    == 1)
        self.positions_x+= -1*(actions    == 2)
        self.positions_y+=  1*(actions    == 2)
        self.positions_y+= -1*(actions    == 3)

        self.positions_x = numpy.clip(self.positions_x, 0, self.size-1)
        self.positions_y = numpy.clip(self.positions_y, 0, self.height-1)

        for e in range(self.envs_count):
            y = self.positions_y[e]
            x = self.positions_x[e]
            
            if self.tunnel[y][x] < 0.0:
                dones[e]       = True
                rewards[e]     = -1.0

            elif self.tunnel[y][x] > 0.0:
                dones[e]       = True
                rewards[e]     = 1.0

            elif self.steps[e] > 2*self.size:
                dones[e]       = True

        return self._update_observations(), rewards, dones, infos

    def render(self, env_id = 0):       
        image = numpy.zeros((3, self.size, self.height))

        for y in range(self.height):
            for x in range(self.size):
                
                if y == self.positions_y[env_id] and x == self.positions_x[env_id]:
                    image[0][x][y] = 1.0
                    image[1][x][y] = 0.0
                    image[2][x][y] = 0.0

                elif self.tunnel[y][x] < 0.0:
                    image[0][x][y] = 0.0
                    image[1][x][y] = 0.0
                    image[2][x][y] = 1.0

                elif self.tunnel[y][x] > 0.0:
                    image[0][x][y] = 0.0
                    image[1][x][y] = 1.0
                    image[2][x][y] = 0.0


        image = numpy.swapaxes(image, 0, 2)

        image = cv2.resize(image, (self.size*8, self.height*8), interpolation = cv2.INTER_NEAREST)
      
        window_name = "ENV - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def _update_observations(self):
        observations    = numpy.zeros((2, self.envs_count))

        observations[0] = self.positions_y.copy()/self.height
        observations[1] = self.positions_x.copy()/self.size

        observations    = numpy.transpose(observations)
       
        return observations

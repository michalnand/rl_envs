import numpy
import gym
import cv2

class TargetsEnv:
 
    def __init__(self, envs_count = 128, size = 32):
        self.envs_count     = envs_count 
        self.size           = size

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(1, self.size, self.size))
        self.action_space 	        = gym.spaces.Discrete(5)

        self.map        = numpy.zeros((self.size, self.size), dtype=numpy.float32)

        self.map[0][0]                          = 1.0
        self.map[0][self.size -1]               = 1.0
        self.map[self.size -1][0]               = 1.0
        self.map[self.size -1][self.size -1]    = 1.0

        self.target_reached = numpy.zeros(4, dtype=numpy.int)
        
        self.reset()

    def reset(self, env_id = -1):        
        if env_id == -1:
            self.positions_x = numpy.zeros((self.envs_count), dtype=int)
            self.positions_y = numpy.zeros((self.envs_count), dtype=int)

            for e in range(self.envs_count):
                self.positions_x[e] = self.size//2 - numpy.random.randint(0, 2)
                self.positions_y[e] = self.size//2 - numpy.random.randint(0, 2)

            self.steps     = numpy.zeros(self.envs_count, dtype=int)
            
        else:
            self.positions_x[env_id]    = self.size//2 - numpy.random.randint(0, 2)
            self.positions_y[env_id]    = self.size//2 - numpy.random.randint(0, 2)
            self.steps[env_id]          = 0

        if env_id != -1:
            obs = self._update_observations()[env_id]
        else:
            obs = self._update_observations()
            
        return obs


    def step(self, actions):
        rewards = numpy.zeros(self.envs_count, dtype=numpy.float32)
        dones   = numpy.zeros(self.envs_count, dtype=bool)

        self.steps+= 1

        actions    = numpy.array(actions)

        self.positions_x+=  1*(actions    == 1)
        self.positions_x+= -1*(actions    == 2)
        self.positions_y+=  1*(actions    == 3)
        self.positions_y+= -1*(actions    == 4)

        self.positions_x = numpy.clip(self.positions_x, 0, self.size-1)
        self.positions_y = numpy.clip(self.positions_y, 0, self.size-1)


        for e in range(self.envs_count):
            y = self.positions_y[e]
            x = self.positions_x[e]

            #one of targets reached, reward 1
            if self.map[y][x] > 0.999:
                dones[e]       = True
                rewards[e]     = 1.0

                #add reached stats
                x = x//(self.size-1)
                y = y//(self.size-1)

                if y == 0 and x == 0:
                    self.target_reached[0]+= 1
                elif y == 0 and x == 1:
                    self.target_reached[1]+= 1
                elif y == 1 and x == 0:
                    self.target_reached[2]+= 1
                elif y == 1 and x == 1:
                    self.target_reached[3]+= 1

            #steps time out
            elif self.steps[e] > 4*self.size:
                dones[e]       = True

        probs = self.target_reached/(self.target_reached.sum() + 0.0000001)
        infos = numpy.repeat(numpy.expand_dims(probs, 0), self.envs_count, axis=0)

        return self._update_observations(), rewards, dones, infos

    def render(self, env_id = 0):       
        image = numpy.zeros((3, self.size, self.size))

        image[2]        = self.map.copy()

        y = self.positions_y[env_id]
        x = self.positions_x[env_id]

        image[0][y][x]  = 1.0

        image = numpy.swapaxes(image, 0, 2)
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_NEAREST)
      
        window_name = "ENV - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def _update_observations(self):
        tmp             = numpy.expand_dims(self.map, 0)
        tmp             = numpy.expand_dims(tmp, 0)

        obs             = numpy.zeros(self.observation_space.shape, dtype=numpy.float32)
        obs[0]          = self.map.copy()
        obs             = numpy.expand_dims(obs, 0)

        observations    = numpy.repeat(obs, self.envs_count, axis=0)

        for e in range(self.envs_count):
            y = self.positions_y[e]
            x = self.positions_x[e]

            observations[e][0][y][x] = 2.0
    
        return observations


if __name__ == "__main__":
    envs_count = 128
    envs = TargetsEnv(envs_count, 32)

    obs = envs.reset()
    print("obs_shape = ", obs.shape)

    while True:
        actions = numpy.random.randint(0, 5, envs_count)
        obs, reward, done, info = envs.step(actions)

        if reward[0] != 0:
            envs.render()
            print(reward[0], info[0])

        for e in range(envs_count):
            if done[e]:
                envs.reset(e)

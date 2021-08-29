import numpy
import gym
import cv2

class TargetsWallEnv:
 
    def __init__(self, envs_count = 128, size = 32):
        self.envs_count     = envs_count 
        self.size           = size

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(1, self.size, self.size))
        self.action_space 	        = gym.spaces.Discrete(5)

        self.map        = numpy.zeros((self.size, self.size), dtype=numpy.float32)

        self.map[0][0]                          = 1.0

        for i in range(self.size):
            self.map[i][self.size//2]       = -1.0

        self.map[self.size -1][self.size -1]    = 1.0

        self.target_reached = numpy.zeros(2, dtype=numpy.int)
        self.fields_visited = numpy.zeros((self.size, self.size), dtype=numpy.int)
        
        self.reset()

    def reset(self, env_id = -1):        
        if env_id == -1:
            self.positions_x = numpy.zeros((self.envs_count), dtype=int)
            self.positions_y = numpy.zeros((self.envs_count), dtype=int)

            for e in range(self.envs_count):
                self.positions_x[e] = 1
                self.positions_y[e] = self.size//2  - numpy.random.randint(0, 2)

            self.steps     = numpy.zeros(self.envs_count, dtype=int)
            
        else:
            self.positions_x[env_id]    = 1
            self.positions_y[env_id]    = self.size//2  - numpy.random.randint(0, 2)
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


            #step on wall
            if self.map[y][x] < 0.0:
                rewards[e] = -0.0

            #one of targets reached
            elif self.map[y][x] > 0.999:
                dones[e]       = True

                if x > self.size//2:
                    rewards[e]              = 1.0
                    self.target_reached[1]+= 1
                else:
                    rewards[e]              = 0.1
                    self.target_reached[0]+= 1

            #steps time out
            elif self.steps[e] > 4*self.size:
                dones[e]       = True

            self.fields_visited[y][y]+= 1

        eps             = 0.0000001

        targets_probs   = (self.target_reached + eps)/(self.target_reached.sum() + eps)
        targets_entropy = (-targets_probs*numpy.log(targets_probs + eps)).sum()
        
        visited_probs   = (self.fields_visited + eps)/(self.fields_visited.sum() + eps)
        visited_entropy = (-visited_probs*numpy.log(visited_probs)).sum()

        info   = {}
        info["targets_probs"]   = targets_probs
        info["targets_entropy"] = targets_entropy
        info["visited_entropy"] = visited_entropy


        infos   = []
        infos.append(info)

        return self._update_observations(), rewards, dones, infos

    def render(self, env_id = 0):       
        image = numpy.zeros((3, self.size, self.size))

        image[2]        = self.map > 0.0
        image[0]        = self.map < 0.0

        y = self.positions_y[env_id]
        x = self.positions_x[env_id]

        image[0][y][x]  = 1.0
        image[1][y][x]  = 1.0
        image[2][y][x]  = 1.0

        image    = numpy.moveaxis(image, 0, 2)

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

            observations[e][0][y][x] = 0.5
    
        return observations


if __name__ == "__main__":
    envs_count = 8
    envs = TargetsWallEnv(envs_count, 32)

    obs = envs.reset()
    print("obs_shape = ", obs.shape)

    while True:
        actions = numpy.random.randint(0, 5, envs_count)
        obs, reward, done, info = envs.step(actions)

        if reward[0] > 0:
            envs.render()
            print(reward[0])

        for e in range(envs_count):
            if done[e]:
                envs.reset(e)

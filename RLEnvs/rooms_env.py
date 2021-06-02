import numpy
import gym
import cv2

class RoomsEnv:
 
    def __init__(self, envs_count = 16, size = 16, grid_size = 8):
        self.envs_count     = envs_count 
        self.size           = size
        self.grid_size      = grid_size

        self.observation_space 		= gym.spaces.Box(low=-1, high=1, shape=(3, self.size, self.size))
        self.action_space 	        = gym.spaces.Discrete(5)

        self._random_seed = 0

        self.maps_initial = self._create_maps()

        self.reset()

    def _create_map(self):
        rooms_colors = []
        
        rooms_colors.append([0.3, 0.3, 0.3])
        rooms_colors.append([0.0, 0.0, 0.5])
        rooms_colors.append([0.0, 0.5, 0.0])
        rooms_colors.append([0.0, 0.5, 0.5])
        rooms_colors.append([0.5, 0.0, 0.0])
        rooms_colors.append([0.5, 0.0, 0.5])
        rooms_colors.append([0.5, 0.5, 0.0])
        

        items_colors = []

        items_colors.append([0.0, 0.0, 1.0])
        items_colors.append([0.0, 0.5, 1.0])
        items_colors.append([0.0, 1.0, 0.0])
        items_colors.append([0.0, 1.0, 0.5])
        items_colors.append([0.0, 1.0, 1.0])
        items_colors.append([0.5, 0.0, 1.0])
        items_colors.append([0.5, 0.5, 1.0])
        items_colors.append([0.5, 1.0, 0.0])
        items_colors.append([0.5, 1.0, 0.5])
        items_colors.append([0.5, 1.0, 1.0])
        items_colors.append([1.0, 0.0, 0.0])
        items_colors.append([1.0, 0.0, 0.5])
        items_colors.append([1.0, 0.0, 1.0])
        items_colors.append([1.0, 0.5, 0.0])
        items_colors.append([1.0, 0.5, 0.5])
        items_colors.append([1.0, 0.5, 1.0])
        items_colors.append([1.0, 1.0, 0.0])
        items_colors.append([1.0, 1.0, 0.5])
        
        rooms_colors      = numpy.array(rooms_colors) 
        items_colors      = numpy.array(items_colors) 


        map    = numpy.zeros((self.grid_size, self.grid_size, 3, self.size, self.size))

        for ry in range(self.grid_size):
            for rx in range(self.grid_size):
                c_idx       = self._rand()%len(rooms_colors)
                c           = rooms_colors[c_idx].reshape(3, 1, 1)
                tmp         = numpy.tile(c, (1, self.size, self.size))

                for _ in range(self.size): 
                    x       = self._rand()%self.size
                    y       = self._rand()%self.size
                    c_idx   = self._rand()%len(items_colors)

                    tmp[0][y][x] = items_colors[c_idx][0]
                    tmp[1][y][x] = items_colors[c_idx][1]
                    tmp[2][y][x] = items_colors[c_idx][2] 

                map[ry][rx] = tmp.copy()

        return map

    def _create_maps(self):
        maps    = numpy.zeros((self.envs_count, self.grid_size, self.grid_size, 3, self.size, self.size))

        map     = self._create_map()
        for e in range(self.envs_count):
            maps[e] = map.copy()

        return maps



    def _rand(self):
        self._random_seed = 1103515245*self._random_seed + 12345
        return self._random_seed//256


    def reset(self, env_id = -1):
        if env_id == -1:
            self.positions = numpy.zeros((self.envs_count, 2), dtype=int)
            for i in range(self.envs_count):
                self.positions[i][0] = self.size//2
                self.positions[i][1] = self.size//2

            self.steps     = numpy.zeros(self.envs_count, dtype=int)
            self.explored  = numpy.zeros((self.envs_count, self.grid_size, self.grid_size), dtype=bool)
            
            
            self.maps      = self.maps_initial.copy()

        else:
            self.positions[env_id][0]   = self.size//2
            self.positions[env_id][1]   = self.size//2
            self.steps[env_id]          = 0
            self.explored[env_id]       = numpy.zeros((self.grid_size, self.grid_size), dtype=bool)

            self.maps[env_id]           = self.maps_initial[env_id].copy()

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

        d_position = numpy.zeros((self.envs_count, 2), dtype=int)

        actions    = numpy.array(actions).reshape((self.envs_count, 1))

        d_position+= [0, 0]*(actions == 0)
        d_position+= [1, 0]*(actions == 1)
        d_position+= [-1, 0]*(actions == 2)
        d_position+= [0, 1]*(actions == 3)
        d_position+= [0, -1]*(actions == 4)

        positions_new = numpy.clip(self.positions + d_position, 0, self.size*self.grid_size-1)

        for e in range(self.envs_count):
            ry = positions_new[e][0]//self.size
            rx = positions_new[e][1]//self.size

            if self.explored[e][ry][rx] == False:
                self.maps[e][ry][rx] = self.maps_initial[e][ry][rx].copy()
            else:
                self.maps[e][ry][rx] = 1.0 - self.maps_initial[e][ry][rx].copy()

        self.positions = positions_new

        max_steps = 4*self.size*self.grid_size*self.grid_size
        
        for e in range(self.envs_count):
            ry = self.positions[e][0]//self.size
            rx = self.positions[e][1]//self.size
            
            if self.explored[e][ry][rx] == False:
                rewards[e]                  = 1.0
                self.explored[e][ry][rx]    = True

                if numpy.sum(self.explored[e]) == self.grid_size*self.grid_size:
                    dones[e]   = True

            if self.steps[e] >= max_steps:
                dones[e] = True

        return self._update_observations(), rewards, dones, infos


    def render(self, env_id = 0):       
        image = self._update_observations()[env_id]
        space = 0

        height = self.grid_size*(space + self.size)
        width  = self.grid_size*(space + self.size)

        image = numpy.zeros((3, height, width))
        
        for ry in range(self.grid_size):
            for rx in range(self.grid_size):
                ys = ry*(self.size + space) + space//2
                xs = rx*(self.size + space) + space//2

                ye = ys + self.size
                xe = xs + self.size

                image[0][ys:ye, xs:xe] = self.maps[env_id][ry][rx][0]
                image[1][ys:ye, xs:xe] = self.maps[env_id][ry][rx][1]
                image[2][ys:ye, xs:xe] = self.maps[env_id][ry][rx][2]

        py      = self.positions[env_id][0]
        px      = self.positions[env_id][1]

        image[0][py][px] = 1.0
        image[1][py][px] = 1.0
        image[2][py][px] = 1.0

        image = numpy.swapaxes(image, 0, 2)

        image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_NEAREST)
      
        window_name = "ENV - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def _update_observations(self):
        observations = numpy.zeros((self.envs_count, 3, self.size, self.size))

        for e in range(self.envs_count):
            room_y  = self.positions[e][0]//self.size
            room_x  = self.positions[e][1]//self.size

            py      = self.positions[e][0]%self.size
            px      = self.positions[e][1]%self.size

            observations[e] = self.maps[e][room_y][room_x].copy()
            observations[e][0][py][px] = 1.0
            observations[e][1][py][px] = 1.0
            observations[e][2][py][px] = 1.0
       
        return observations

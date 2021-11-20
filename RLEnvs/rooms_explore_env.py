import numpy
import gym
import cv2

class RoomsExploreEnv:
    def __init__(self, envs_count = 4, grid_size = 8, room_size = 16, upscale = 6):
        self.envs_count         = envs_count 
        self.grid_size          = grid_size
        self.room_size          = room_size
        self.upscale            = upscale
        

        self.observation_space  = gym.spaces.Box(low=-1, high=1, shape=(3, self.room_size*self.upscale, self.room_size*self.upscale))
        self.action_space 	    = gym.spaces.Discrete(5)

        self._random_seed = 0
        

        self.map_initial, self.points = self._create_map()

        self.visiting_counts    = numpy.zeros(self.grid_size*self.grid_size, dtype=int)
        self.explored_rooms     = 0

        self.reset()

    def _create_map(self):

        colors = [0.1, 0.2, 0.4, 0.6]

        rooms_colors = []
 
        for r in colors:
            for g in colors:
                for b in colors:
                    rooms_colors.append([r, g, b])

        rooms_colors = numpy.array(rooms_colors) 

        map     = numpy.zeros((self.grid_size, self.grid_size, 3, self.room_size, self.room_size))
        points  = numpy.zeros((self.grid_size, self.grid_size, 2), dtype=int)

        color_idx = 0
        for ry in range(self.grid_size):
            for rx in range(self.grid_size):
                tmp         = rooms_colors[color_idx].reshape(3, 1, 1)
                map[ry][rx] = numpy.tile(tmp, (1, self.room_size, self.room_size))

                color_idx = (color_idx+1)%len(rooms_colors)

                #create point at random position
                x       = self._rand()%self.room_size
                y       = self._rand()%self.room_size
 
                points[ry][rx][0] = x
                points[ry][rx][1] = y

        return map, points


    def _rand(self):
        self._random_seed = 1103515245*self._random_seed + 12345
        return self._random_seed//256

    def reset(self, env_id = -1):
        if env_id == -1:
            self.positions = numpy.zeros((self.envs_count, 2), dtype=int)
            for e in range(self.envs_count):
                self.positions[e][0] = self.room_size//2
                self.positions[e][1] = self.room_size//2
            
            self.points_active = numpy.ones((self.envs_count, self.grid_size, self.grid_size), dtype=bool)

            self.steps  = numpy.zeros(self.envs_count, dtype=int)

            self.score_sum = numpy.zeros(self.envs_count, dtype=numpy.float32)
            
        else:
            self.positions[env_id][0]   = self.room_size//2
            self.positions[env_id][1]   = self.room_size//2
            self.points_active[env_id]  = numpy.ones((self.grid_size, self.grid_size), dtype=bool)
            self.steps[env_id]          = 0
            self.score_sum[env_id]      = 0

        if env_id != -1:
            obs = self._update_observations()[env_id]
        else:
            obs = self._update_observations()
            
        return obs


    def step(self, actions):
        rewards = numpy.zeros(self.envs_count, dtype=numpy.float32)
        dones   = numpy.zeros(self.envs_count, dtype=bool)
        infos   = []

        self.steps+= 1

        d_position = numpy.zeros((self.envs_count, 2), dtype=int)

        actions    = numpy.array(actions).reshape((self.envs_count, 1))

        d_position+= [0, 0]*(actions == 0)
        d_position+= [1, 0]*(actions == 1)
        d_position+= [-1, 0]*(actions == 2)
        d_position+= [0, 1]*(actions == 3)
        d_position+= [0, -1]*(actions == 4)

        self.positions = numpy.clip(self.positions + d_position, 0, self.room_size*self.grid_size - 1)

        max_steps = 4*self.room_size*self.grid_size*self.grid_size
        
        for e in range(self.envs_count):
            ry      = self.positions[e][0]//self.room_size
            rx      = self.positions[e][1]//self.room_size
            ofs_y   = self.positions[e][0]%self.room_size
            ofs_x   = self.positions[e][1]%self.room_size

            #points collected
            if self.points[ry][rx][0] == ofs_y and self.points[ry][rx][1] == ofs_x and self.points_active[e][ry][rx]:
                rewards[e]        = 1.0
                self.score_sum[e]+= 1

                #collect point
                self.points_active[e][ry][rx]   = False

            #all points collected
            if numpy.sum(self.points_active[e]) == 0:
                dones[e] = True
            
            #max steps reached
            if self.steps[e] >= max_steps:
                dones[e] = True

            if self.score_sum[e] > self.explored_rooms:
                self.explored_rooms = int(self.score_sum[e])


            room_id = ry*self.grid_size + rx
            self.visiting_counts[room_id]+= 1

            visiting_stats = self.visiting_counts/self.visiting_counts.sum()
            visiting_stats = numpy.around(visiting_stats, decimals=3)
 
            info = {}
            info["room_id"]                 = room_id
            info["explored_rooms"]          = numpy.sum(self.visiting_counts != 0)
            info["rooms_visiting_stats"]    = visiting_stats
            infos.append(info)

        return self._update_observations(), rewards, dones, infos


    def render(self, env_id = 0):
        obs     = self._update_observations()[env_id]

        #image   = numpy.swapaxes(obs, 0, 2)
        image   = numpy.moveaxis(obs, 0, 2)

        image   = cv2.resize(image, (512, 512), interpolation = cv2.INTER_NEAREST)
      
        window_name = "ENV - " + self.__class__.__name__
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)


    def _update_observations(self):
        observations = numpy.zeros((self.envs_count, 3, self.room_size, self.room_size), dtype=numpy.float32)

        text_print   = numpy.zeros((self.envs_count, self.room_size*self.upscale, self.room_size*self.upscale), dtype=numpy.float32)

        
        for e in range(self.envs_count):
            room_y  = self.positions[e][0]//self.room_size
            room_x  = self.positions[e][1]//self.room_size

            #player position
            py0     = self.positions[e][0]%self.room_size
            px0     = self.positions[e][1]%self.room_size

            #point position
            py1     = self.points[room_y][room_x][0]
            px1     = self.points[room_y][room_x][1]

            observations[e] = self.map_initial[room_y][room_x].copy()
 
            #add point if not taken, gray
            if self.points_active[e][room_y][room_x]:
                observations[e][0][py1][px1] = 0.8
                observations[e][1][py1][px1] = 0.8
                observations[e][2][py1][px1] = 0.8

            #add player position, white 
            observations[e][0][py0][px0] = 1.0
            observations[e][1][py0][px0] = 1.0
            observations[e][2][py0][px0] = 1.0

            font = cv2.FONT_HERSHEY_PLAIN
            text = str(int(self.score_sum[e]))
            cv2.putText(text_print[e], text,(16,16), font, 1, (255,255,255), 1, cv2.LINE_AA)

        text_print = text_print/256.0

        #upsample
        observations = numpy.repeat(observations, self.upscale, axis=2)
        observations = numpy.repeat(observations, self.upscale, axis=3)

        #add score text
        for e in range(self.envs_count):
            observations[e][0]+= text_print[e].copy()
            observations[e][1]+= text_print[e].copy()
            observations[e][2]+= text_print[e].copy()

        observations = numpy.clip(observations, 0.0, 1.0)

        return observations

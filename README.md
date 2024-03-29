# rl_envs

rl environments

# install

requires : gym, numpy, cv2

```bash
pip3 install -e .
```

**usage**

```python
import RLEnvs
import time
import numpy

envs_count = 128

#create parallel envs
env             = RLEnvs.RoomsEnv(envs_count = envs_count)

#actions count
actions_count   = env.action_space.n

fps = 0.0
k   = 0.01

while True:

    time_start = time.time()
    #random actions for all 128 envs
    actions = numpy.random.randint(actions_count, size=envs_count)
    #env step
    states, rewards, dones, _ = env.step(actions)
    time_stop = time.time()
    
    #reset if done
    for i in range(len(dones)):
        if dones[i]:
            states = env.reset(i)

    #fps smoothing
    fps = (1.0 - k)*fps + k/(time_stop - time_start)

    print("fps = ", round(fps, 2), rewards[0])

    #render env ID=0
    env.render(0)

```


# environments 

## rooms

fast parallel rooms exploring env

![image](doc/rooms.gif)

- observation : envs_count, 3, 16, 16 (set of RGB images)
- action      : 5 : 4 for move, 1 do nothig


## tunnel

deadly tunnel, reward is on its end

![image](doc/tunnel.gif)

- observation : envs_count, 2 : positionof agent, normalised into (0, 1)
- action      : 5 : 4 for move, 1 do nothig

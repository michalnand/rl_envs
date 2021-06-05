import RLEnvs
import time
import numpy

envs_count = 128

#create parallel envs
env             = RLEnvs.TunnelEnv(envs_count = envs_count)

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

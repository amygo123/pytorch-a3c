import time
from collections import deque

import torch
import torch.nn.functional as F
from EnvAPI import Env
from envs import create_atari_env
from model import ActorCritic


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)
    torch.save(shared_model.state_dict(), 't.pkl')

    env = Env(args.seed + rank)
    model = ActorCritic(1, env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    # env.visual()

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=500)
    episode_length = 0
    while True:

        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())


        with torch.no_grad():
            value, logit = model((state.unsqueeze(0)).type(torch.FloatTensor))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        print(action)

        state, reward, done = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            # env.visual()
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            time.sleep(60)

        state = torch.from_numpy(state)

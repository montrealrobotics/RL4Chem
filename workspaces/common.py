import gym
import utils
import numpy as np

def make_agent(envs, device, cfg):
    if cfg.agent == 'ppo':
        from agents.ppo import PpoAgent

        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        num_states = np.array(envs.single_observation_space.shape).prod()
        num_actions = envs.single_action_space.n

        print("envs.single_observation_space.space", num_states)
        print("envs.single_action_space.n", num_actions)

        agent = PpoAgent(device, num_states, num_actions)
                            
    else:
        raise NotImplementedError

    return agent
        
def make_env(cfg):
    if 'gym' in cfg.benchmark:
        import gym
        def get_gymenv(cfg):
            def thunk():
                env = gym.make(cfg.id) 
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.seed(seed=cfg.seed)
                env.observation_space.seed(cfg.seed)
                env.action_space.seed(cfg.seed)
                return env 
            return thunk

        return gym.vector.SyncVectorEnv([get_gymenv(cfg) for i in range(cfg.num_envs)]), gym.vector.SyncVectorEnv([get_gymenv(cfg) for i in range(cfg.num_envs)]) 
    
    else:
        raise NotImplementedError
import os
import shutil
import threading
import time
from os.path import basename
from typing import Optional

import gym
import numpy as np
import torch
from rl_games.common import vecenv, env_configurations
from rl_games.algos_torch import  model_builder


class BasePlayer:
    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.env_config = self.config.get('env_config', {})
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)
        if self.env_info is None:
            self.env = vecenv.create_vec_env(self.env_name, self.config['num_actors'], **self.env_config)
            self.env_info = self.env.get_env_info()

        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']
        self.num_agents = self.env_info.get('agents', 1)

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get('central_value_config') is not None
        # self.device_name = self.config.get('device_name', 'cuda')
        self.device_name = self.config.get('device_name', 'cpu')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)
        self.is_determenistic = self.player_config.get('determenistic', True)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
        self.device = torch.device(self.device_name)

        self.evaluation = self.player_config.get("evaluation", False)

        # if we run player as evaluation worker this will take care of loading new checkpoints
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[str] = None
        self.checkpoint_mutex = threading.Lock()

        if self.evaluation and self.dir_to_monitor is not None:
            self.eval_checkpoint_dir = os.path.join(self.dir_to_monitor, "eval_checkpoints")
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler
            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(self.file_events, self.dir_to_monitor, recursive=False)
            self.file_observer.start()

        player_init_callbacks = params.get("player_init_callbacks", [])
        for player_init_callback in player_init_callbacks:
            player_init_callback(params, self)

        self.player_reset_callbacks = params.get("player_reset_callbacks", [])
        self.player_step_callbacks = params.get("player_step_callbacks", [])
        self.player_before_action_callbacks = params.get("player_before_action_callbacks", [])
        self.player_action_callbacks = params.get("player_action_callbacks", [])

    def wait_for_checkpoint(self):
        if self.dir_to_monitor is None:
            return

        if self.checkpoint_to_load is None:
            print("Waiting for first eval checkpoint...")
        while True:
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    break
            time.sleep(0.1)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self):
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(f"Evaluation: loading new checkpoint {self.checkpoint_to_load}...")
                self.restore(self.checkpoint_to_load)
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this
            eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
            shutil.copyfile(path, eval_checkpoint_path)

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_determenistic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_determenistic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            #print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            for cb in self.player_reset_callbacks:
                cb(obses)

            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if n % 100 == 0:
                    self.maybe_load_new_checkpoint()

                for cb in self.player_before_action_callbacks:
                    cb(obses)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses, is_determenistic)

                for cb in self.player_action_callbacks:
                    action = cb(action)

                obses, r, done, info = self.env_step(self.env, action)
                for cb in self.player_step_callbacks:
                    cb(obses, r, done, info)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count,
                                  'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size

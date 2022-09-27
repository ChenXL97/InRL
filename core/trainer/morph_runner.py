from rl_games.common import experiment

from rl_games.common.algo_observer import DefaultAlgoObserver

from rl_games.torch_runner import Runner


class MorphRunner(Runner):
    def __init__(self, algo_observer=None):
        super(MorphRunner, self).__init__(algo_observer)
        self.load_path = None

    def _expand_cfg(self):
        if 'features' not in self.config:
            self.config['features'] = {}
        self.config['features']['observer'] = self.algo_observer
        num_envs = self.morph_cfg['num_morphs'] * self.morph_cfg['envs_per_morph']
        self.morph_cfg['num_envs'] = num_envs
        self.config['env_config'] = {'morph_cfg': self.morph_cfg}
        self.config['num_actors'] = num_envs
        self.config['model_name'] = self.morph_cfg['model_name']
        self.config['minibatch_size'] = int(num_envs * self.config['horizon_length'] * self.config['minibatch_rate'])

    def set_morph_cfg(self, morph_cfg):
        self.morph_cfg = morph_cfg

    def run_train(self):
        print('Started to train')
        if self.algo_observer is None:
            self.algo_observer = DefaultAlgoObserver()

        if self.exp_config:
            self.experiment = experiment.Experiment(self.default_config, self.exp_config)
            exp_num = 0
            exp = self.experiment.get_next_config()
            while exp is not None:
                exp_num += 1
                print('Starting experiment number: ' + str(exp_num))
                self.reset()
                self.load_config(exp)
                self._expand_cfg()
                agent = self.algo_factory.create(self.algo_name, base_name='run', config=self.config)
                self.experiment.set_results(*agent.train())
                exp = self.experiment.get_next_config()
        else:
            self.reset()
            self.load_config(self.default_config)
            self._expand_cfg()
            agent = self.algo_factory.create(self.algo_name, base_name='run', config=self.config)
            if self.load_check_point and (self.load_path is not None):
                agent.restore(self.load_path)
            return agent.train()

    def run(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None:
            if len(args['checkpoint']) > 0:
                self.load_path = args['checkpoint']

        if args['play']:
            print('Started to play')
            self.load_config(self.default_config)
            self._expand_cfg()
            player = self.create_player()
            if self.load_path:
                player.restore(self.load_path)
            return player.run()
        else:
            return self.run_train()

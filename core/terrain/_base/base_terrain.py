class BaseTerrain(object):
    def __init__(self, task):
        self.task = task
        self.cfg = task.cfg
        self.gym = task.gym
        self.sim = task.sim

    def set_agent_init(self):
        raise NotImplementedError

    def create_terrain(self):
        raise NotImplementedError

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class DebugPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def after_training_iteration(self, strategy, **kwargs):
        strategy.stop_training()

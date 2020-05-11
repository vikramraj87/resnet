class CometNotifier:
    def __init__(self):
        pass

    def epoch_started(self, epoch):
        pass

    def step_started(self, step):
        pass

    def train_step_completed(self, loss):
        pass

    def validation_step_completed(self, loss, accuracy):
        pass

    def step_completed(self, step: int):
        pass

    def epoch_completed(self,
                        epoch,
                        step,
                        model_state_dict,
                        optim_state_dict):
        pass
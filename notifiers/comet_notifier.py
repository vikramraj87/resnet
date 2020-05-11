from comet_ml import Experiment
import torch


class CometNotifier:
    def __init__(self,
                 experiment: Experiment,
                 tmp_path):
        self.experiment = experiment
        self.file_path = tmp_path

        # self.step_cache = []
        # self.train_loss = []
        # self.val_loss = []
        # self.val_acc = []

    def epoch_started(self, epoch):
        self.experiment.set_epoch(epoch)

    def step_started(self, step, epoch):
        self.experiment.set_step(step)
        # self.step_cache.append(f"{epoch}_{step}")

    def train_step_completed(self, loss):
        with self.experiment.train():
            self.experiment.log_metric("loss", loss)

        # self.train_loss.append(loss)

    def validation_step_completed(self, loss, accuracy):
        with self.experiment.validate():
            self.experiment.log_metric("loss", loss)
            self.experiment.log_metric("accuracy", accuracy)

        # self.val_loss.append(loss)
        # self.val_acc.append(accuracy)

    def epoch_completed(self,
                        epoch,
                        step,
                        model_state_dict,
                        optim_state_dict):
        self.save_checkpoint(epoch,
                             step,
                             model_state_dict,
                             optim_state_dict)
        self.experiment.log_epoch_end(epoch)

    def save_checkpoint(self,
                        epoch,
                        step,
                        model_state_dict,
                        optim_state_dict,
                        is_best=False):
        state = {
            # "steps": self.step_cache,
            # "train_loss": self.train_loss,
            # "val_loss": self.val_loss,
            # "val_accuracy": self.val_acc,
            "epoch": epoch,
            "step": step,
            "state_dict": model_state_dict,
            "optim_dict": optim_state_dict
        }

        torch.save(state, self.file_path)

        # Copy binary file as asset to comet.ml
        file_name = "best" if is_best else "checkpoint"
        fp = open(self.file_path, "rb")
        self.experiment.log_asset(fp, file_name=f"{file_name}.pt")
        fp.close()

    # @staticmethod
    # def epoch_step(step_str: str) -> (int, int):
    #     epoch_str, step_s = step_str.split("_")
    #     return int(epoch_str), int(step_s)
    #
    # @staticmethod
    # def step_key(epoch: int, step: int) -> str:
    #     return f"{epoch}_{step}"
    #
    # def resume_state(self, state):
    #     self._update(state["steps"],
    #                  state["train_loss"],
    #                  state["val_loss"],
    #                  state["val_accuracy"])
    #
    # def _update(self,
    #             step_cache,
    #             train_loss,
    #             val_loss,
    #             val_acc):
    #     if len(step_cache) == 0:
    #         return
    #
    #     self.step_cache = []
    #     self.train_loss = []
    #     self.val_loss = []
    #     self.val_acc = []
    #
    #     first_step = step_cache[0]
    #     curr_epoch, _ = self.epoch_step(first_step)
    #
    #     # Update the previous state
    #     self.experiment.set_epoch(curr_epoch)
    #     for step_str, t_l, v_l, v_a in zip(step_cache,
    #                                        train_loss,
    #                                        val_loss,
    #                                        val_acc):
    #         self.step_cache.append(step_str)
    #         epoch, step = self.epoch_step(step_str)
    #
    #         if epoch > curr_epoch:
    #             self.experiment.log_epoch_end(curr_epoch)
    #             curr_epoch = epoch
    #             self.experiment.set_epoch(curr_epoch)
    #
    #         self.experiment.set_step(step)
    #
    #         self.train_loss.append(t_l)
    #         with self.experiment.train():
    #             self.experiment.log_metric("loss", t_l)
    #
    #         self.val_loss.append(v_l)
    #         self.val_acc.append(v_a)
    #         with self.experiment.validate():
    #             self.experiment.log_metric("loss", v_l)
    #             self.experiment.log_metric("accuracy", v_a)


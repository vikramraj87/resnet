class StdoutNotifier:
    def __init__(self,
                 report_interval=100):
        self.report_interval = report_interval

        self.running_train_loss = 0
        self.running_val_loss = 0
        self.running_val_acc = 0

    def reset(self):
        self.running_train_loss = 0
        self.running_val_loss = 0
        self.running_val_acc = 0

    @staticmethod
    def epoch_started(epoch):
        print(f"Epoch: {epoch}")

    def train_step_completed(self, loss):
        self.running_train_loss += loss

    def validation_step_completed(self, loss, accuracy):
        self.running_val_loss += loss
        self.running_val_acc += accuracy

    def step_completed(self, step: int):
        curr_progress = step % self.report_interval
        if curr_progress != 0:
            print(f"\r{curr_progress}/{self.report_interval}", end="")
            return

        train_loss = self.running_train_loss / self.report_interval
        val_loss = self.running_val_loss / self.report_interval
        val_acc = self.running_val_acc / self.report_interval

        out = f"\r[{step:2}] Train loss: {train_loss:.2f}; "
        out += f"Validation loss: {val_loss:.2f}; "
        out += f"Validation accuracy: {val_acc:.2f}"
        print(out)

        self.reset()

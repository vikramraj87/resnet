class TrainRunner:
    def __init__(self,
                 trainer,
                 train_loader,
                 n_epochs,
                 device,
                 train_split=0.8,
                 epochs_completed=0,
                 steps_completed=0,
                 notifiers=None):
        self.trainer = trainer
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.device = device
        self.epochs_completed = epochs_completed
        self.steps_completed = steps_completed
        self.notifiers = notifiers

        batch_size = train_loader.batch_size
        self.n_train = round(train_split * batch_size)
        self.n_test = batch_size - self.n_train

    def __call__(self):
        step = self.steps_completed
        for epoch in range(self.epochs_completed+1, self.n_epochs+1):
            self._notify("epoch_started",
                         epoch=epoch)

            for (images, labels) in self.train_loader:
                step += 1
                self._notify("step_started",
                             step=step,
                             epoch=epoch)

                x = images[:self.n_train].to(self.device)
                y = labels[:self.n_train].to(self.device)

                loss = self.trainer.train_step(x, y)
                self._notify("train_step_completed",
                             loss=loss)

                val_x = images[self.n_train:].to(self.device)
                val_y = labels[self.n_train:].to(self.device)
                val_loss, val_acc = self.trainer.validation_step(val_x, val_y)
                self._notify("validation_step_completed",
                             loss=val_loss,
                             accuracy=val_acc)
                self._notify("step_completed",
                             epoch=epoch,
                             step=step,
                             model_state_dict=self.trainer.model.state_dict(),
                             optim_state_dict=self.trainer.optimizer.state_dict())

            model_state_dict = self.trainer.model.state_dict()
            optim_state_dict = self.trainer.optimizer.state_dict()
            self._notify("epoch_completed",
                         epoch=epoch,
                         step=step,
                         model_state_dict=model_state_dict,
                         optim_state_dict=optim_state_dict)

    def _notify(self, action: str, *args, **kwargs):
        if self.notifiers is None:
            return

        if not isinstance(self.notifiers, list):
            notifiers = [self.notifiers]
        else:
            notifiers = self.notifiers

        for notifier in notifiers:
            func = getattr(notifier, action, None)
            if callable(func):
                func(*args, **kwargs)
from keras.callbacks import ModelCheckpoint


class SquadTrainer:
    def __init__(self, model, train_generator, epoch, dev_generator, save_path):
        self.model = model
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epoch = epoch
        self.callbacks = [ModelCheckpoint(save_path)]

    def run(self):
        return self.model.fit_generator(
            generator=self.train_generator, epochs=self.epoch, validation_data=self.dev_generator or None,
            steps_per_epoch=len(self.train_generator), validation_steps=len(self.dev_generator),
            callbacks=self.callbacks)

    def add_callback(self, callback):
        self.callbacks.append(callback)

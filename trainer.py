class SquadTrainer:
    def __init__(self, model, train_generator, epoch, dev_generator=None):
        self.model = model
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epoch = epoch

    def run(self):
        self.model.fit_generator(
            generator=self.train_generator, epochs=self.epoch, validation_data=self.dev_generator)

from unittest import TestCase
from unittest.mock import MagicMock

from trainer import SquadTrainer


class TestSquadTrainer(TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_generator = MagicMock()

    def test_init(self):
        epoch = 100
        trainer = SquadTrainer(self.mock_model, self.mock_generator, epoch)
        self.assertEqual(trainer.epoch, epoch)

    def test_run(self):
        epoch = 100
        trainer = SquadTrainer(self.mock_model, self.mock_generator, epoch)
        self.assertEqual(trainer.epoch, epoch)

        trainer.run()
        self.mock_model.fit_generator.assert_called_with(
            generator=self.mock_generator, epochs=epoch, validation_data=None)

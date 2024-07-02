import numpy as np


class LossTracker:
    def __init__(self, *loss_names):
        self.loss_history = {k: [] for k in loss_names}
        self._batch_loss_history = {k: [] for k in loss_names}

    def batch_update(self, **losses):
        for k, v in losses.items():
            self._batch_loss_history[k].append(v)

    def epoch_update(self):
        self._update_average_losses()
        self._batch_loss_history = {k: [] for k in self._batch_loss_history.keys()}

    def _update_average_losses(self):
        for k, v in self._batch_loss_history.items():
            self.loss_history[k].append(np.mean(v))

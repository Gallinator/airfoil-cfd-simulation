import numpy as np


class LossTracker:
    """
    Helper class to track training losses across epochs and batches.
    batch_update() must be called at the end of each batch, epoch_update() must be called at the end of each epoch.
    loss history contains the batch average losses of each epoch.
    Supports multiple losses
    """

    def __init__(self, *loss_names):
        self.loss_history = {k: [] for k in loss_names}
        self._batch_loss_history = {k: [] for k in loss_names}

    def batch_update(self, **losses):
        """
        :param losses: the actual losses. kwargs must use the same names as the ones used in the constructor
        """
        for k, v in losses.items():
            self._batch_loss_history[k].append(v)

    def epoch_update(self):
        self._update_average_losses()
        self._batch_loss_history = {k: [] for k in self._batch_loss_history.keys()}

    def _update_average_losses(self):
        for k, v in self._batch_loss_history.items():
            self.loss_history[k].append(np.mean(v))

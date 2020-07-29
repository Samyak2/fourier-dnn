import tensorflow as tf

class PSNR(tf.keras.metrics.Metric):

  def __init__(self, name='PSNR', **kwargs):
    super(PSNR, self).__init__(name=name, **kwargs)
    self.last_score = None

  def update_state(self, y_true, y_pred):
    self.last_score = tf.image.psnr(y_true, y_pred, 1.0)

  def result(self):
    return self.last_score
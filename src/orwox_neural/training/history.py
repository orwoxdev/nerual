class History:
    def __init__(self):
        self.history = {
            'loss': [],
            'accuracy': [],
            'grad_magnitude': []
        }

    def add(self, loss, accuracy=None, grad_magnitude=None):
        self.history['loss'].append(float(loss))
        if accuracy is not None:
            self.history['accuracy'].append(float(accuracy))
        if grad_magnitude is not None:
            self.history['grad_magnitude'].append(float(grad_magnitude))

    def get(self):
        return self.history

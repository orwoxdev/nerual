import numpy as np
from .history import History
from ..core.autograd import Tensor
from ..utils.metrics import accuracy_score

class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = History()

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0

            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            num_batches = (n_samples + batch_size - 1) // batch_size

            for i in range(0, n_samples, batch_size):
                batch_X = Tensor(X[i:i+batch_size])
                batch_y = Tensor(y[i:i+batch_size])

                # Forward
                self.model.train()
                preds = self.model.forward(batch_X)
                loss = self.loss_fn(preds, batch_y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data * batch_X.data.shape[0]

                # Accuracy calculation
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    batch_acc = accuracy_score(np.argmax(batch_y.data, axis=1), np.argmax(preds.data, axis=1))
                else:
                    batch_acc = accuracy_score(batch_y.data > 0.5, preds.data > 0.5)
                epoch_acc += batch_acc * batch_X.data.shape[0]

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            # Grad magnitude for history
            grad_mag = np.mean([np.linalg.norm(p.grad) for p in self.model.parameters()])
            self.history.add(epoch_loss, epoch_acc, grad_mag)

            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}")

            if validation_data:
                val_X, val_y = validation_data
                val_loss, val_acc = self.evaluate(val_X, val_y)
                print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        return self.history

    def evaluate(self, X, y):
        self.model.eval()
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        preds = self.model.forward(X_tensor)
        loss = self.loss_fn(preds, y_tensor)

        if len(y.shape) > 1 and y.shape[1] > 1:
            acc = accuracy_score(np.argmax(y, axis=1), np.argmax(preds.data, axis=1))
        else:
            acc = accuracy_score(y > 0.5, preds.data > 0.5)

        return float(loss.data), float(acc)

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class LRFinder:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.loss = []
        self.lrate = []

    def run(self, dataloader, min_lr=1e-7, max_lr=1., beta=0.98):
        lr_factor = (max_lr / min_lr) ** (1 / (len(dataloader)-1))
        
        best_loss = 0.0
        batch_loss_avg = 0.0
        
        lr = min_lr
        self.optimizer.param_groups[0]["lr"] = lr

        self.model.train()
        for b, batch_data in tqdm(enumerate(dataloader)):
            x_batch, y_batch = batch_data
            x_batch, y_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
            )

            self.optimizer.zero_grad()
            y_pred = self.model(x_batch).squeeze()

            batch_loss = self.criterion(y_pred, y_batch)
            batch_loss_avg = beta * batch_loss_avg + (1 - beta) * batch_loss.item()
            batch_loss_smooth = batch_loss_avg / (1 - beta**(b+1))

            if b > 0 and batch_loss_smooth > 4 * best_loss:
                print("Stopping early.")
                return self.lr_with_steepest_gradient(self.lrate, self.loss)

            if batch_loss_smooth < best_loss or b == 0:
                best_loss = batch_loss_smooth

            self.loss.append(batch_loss_smooth)
            self.lrate.append(lr)
            
            batch_loss.backward()
            self.optimizer.step()

            lr *= lr_factor
            self.optimizer.param_groups[0]["lr"] = lr

        return self.lr_with_steepest_gradient(self.lrate, self.loss)

    def plot(self):
        plt.plot(self.lrate[10:-10], self.loss[10:-10])
        plt.xscale("log")
        plt.xlabel("Learning Rate (log10)")
        plt.ylabel("Loss")
        
    def lr_with_steepest_gradient(self, lrate, loss):
        gradients = np.gradient(loss)
        steepest_gradient_idx = np.argmin(gradients)
        return lrate[steepest_gradient_idx]


import matplotlib.pyplot as plt
from MPWAE2           import MPWAE2
from torch            import device, no_grad, amp, save
from torch.optim      import Adam
from torch.nn         import MSELoss
from torch.utils.data import DataLoader


# region TrainingManager class
class TrainingManager:
    def __init__(self,
            model: MPWAE2,
            optimizer: Adam,
            loss: MSELoss, # per project requisite, other options are also viable
            train_loader: DataLoader,
            val_loader: DataLoader,
            dev: device):
        self.model        = model
        self.optimizer    = optimizer
        self.loss         = loss
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = dev

        # Learning rate scheduler
        # ? Uncomment this if using a different Optimizer rather than Adam/W
        # ? SGD benefits a lot from having a scheduler, Adam doesn't really need it
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #   self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        #)

        # training history for plt chart
        self.train_losses   = []
        self.val_losses     = []
        #self.learning_rates = []  # in case learning rate scheduler active


    def _train_epoch(self) -> float:
        self.model.train()

        total_loss  = 0.0
        num_batches = len(self.train_loader)
        dev_type    = self.device.type

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with amp.autocast(device_type=dev_type):
                output = self.model(batch)
                loss   = self.loss(output, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            # TODO: needed if unstable / exploding gradients are an issue
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

            # current progress in output cell (every 20 batches)
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')

        return total_loss / num_batches


    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        dev_type   = self.device.type

        with no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                with amp.autocast(dev_type):
                    output = self.model(batch)
                    loss   = self.loss(output, batch)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)


    def train(self,
            epochs: int,
            patience: int = 15,
            save_path: str = None) -> dict:
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate()

            # Update learning rate
            # self.scheduler.step(val_loss)
            # current_lr = self.optimizer.param_groups[0]['lr']

            # update loss history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            # self.learning_rates.append(current_lr)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'\tTrain Loss: {train_loss:.6f}')
            print(f'\tVal Loss: {val_loss:.6f}')
            # print(f'\tLR: {current_lr:.2e}')

            # early stop and save
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = patience_counter -1 \
                    if patience_counter > 0            \
                    else 0
                best_model_state = self.model.state_dict().copy()
                print(f'New best model with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                print(f'  Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if save_path:
                save(best_model_state, save_path)
                print(f'Best model saved to {save_path}')

        return {
            'train_losses':   self.train_losses,
            'val_losses':     self.val_losses,
            #'learning_rates': self.learning_rates,
            'best_val_loss':  best_val_loss
        }


    def plot_training(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 5), dpi=100)
        plt.title('Model train & val losses')

        plt.xlabel('Epoch')
        plt.ylabel('Losses')

        plt.plot(epochs, self.train_losses, label='Train Loss', alpha=0.5)
        plt.plot(epochs, self.val_losses,   label='Val Loss',   alpha=0.5)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()
# endregion
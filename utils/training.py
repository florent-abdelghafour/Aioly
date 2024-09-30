import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience=10, delta_factor=0.0000025, save_path=None):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.last_loss = np.Inf
        
        self.delta_factor = delta_factor
        
        if save_path is None:
            rel_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_path = os.path.join(rel_dir, 'best_model.pth')
        else:
            self.save_path=save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        # Compute the adaptive delta
        delta = self.delta_factor * self.last_loss if self.last_loss < np.Inf else 0
        if val_loss < self.last_loss - delta:
            self.counter = 0        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, epoch)
        
        self.last_loss=val_loss
      

    def save_checkpoint(self, val_loss, model, epoch):

        checkpoint_path = f'{self.save_path}_best_epoch_{epoch}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss
            
            
            
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None, save_interval=10,early_stop=True,classification = False,scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
 
    early_stopping = EarlyStopping(save_path=save_path) if early_stop and save_path else None
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    val_f1_scores = []
    
    if classification :
        min_val_loss = np.inf
    else :
        min_val_loss = [0]

    for epoch in range(num_epochs):
        model.train()
        if classification:
            running_loss = torch.zeros(1, device=device)
        else :
             running_loss = torch.zeros(model.out_dims, device=device)
        for inputs, targets in train_loader:
            
            inputs = inputs.to(device,non_blocking=True).float()
            targets = targets.to(device,non_blocking=True).float()
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs[:,None])  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            torch.mean(loss).backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.mean(dim=0) * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        if scheduler :
            scheduler.step(epoch_loss)
        train_losses.append(epoch_loss.detach().cpu())       
           
        # Validation loop
        model.eval()
        if classification:
            val_loss = torch.zeros(1, device=device)
        else:
            val_loss = torch.zeros(model.out_dims, device=device)
        out = []
        tar = []
        with torch.no_grad():
            for inputs, targets in val_loader:
               
                inputs = inputs.to(device,non_blocking=True).float()
                targets = targets.to(device,non_blocking=True).float()
                outputs = model(inputs[:,None])
                           
                loss = criterion(outputs, targets) 
                val_loss += loss.mean(dim=0) * inputs.size(0)
                
                out.append(outputs.detach().cpu())
                tar.append(targets.detach().cpu())


        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss.detach().cpu())    
        
        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)
        r2_scores = []
        f1_scores = []
        if not classification:

            R2 = [torcheval.metrics.R2Score() for _ in range(model.out_dims)]

            for i in range(model.out_dims):
                R2[i].update(all_targets[:, i], all_outputs[:, i])
                r2_score = R2[i].compute().item()
                r2_scores.append(r2_score)

            val_r2_scores.append(r2_scores)
        else :

            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(all_targets,dim=1), torch.argmax(all_outputs,dim=1))
            f1_scores = F1.compute()
            val_f1_scores.append(f1_scores)



        train_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(epoch_loss)])
        val_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(val_loss)])
        r2_score_str = ', '.join([f'y {i}: {score:.4f}' for i, score in enumerate(r2_scores)])
        if classification :
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str}| F1 Score: {f1_scores}')
        else :
            print(f'Epoch {epoch+1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | R2 Scores: {r2_score_str}')

        if early_stop:
            early_stopping(val_loss.mean(), model, epoch + 1)
            if early_stopping.early_stop:
                break

        # save best model via validation loss and 30% of total epochs , and save path with best name
        #update the min_val_loss

        if classification:
            if save_path and val_loss.mean() < min_val_loss and (epoch + 1) > num_epochs*0.1 :
                min_val_loss = val_loss.mean()
                final_save_path = save_path + f'_final.pth'
                torch.save(model.state_dict(), final_save_path)
                print(f'Model saved at epoch {epoch + 1} to {final_save_path}')
        else :
            val = np.array(r2_scores)
            if save_path and (val.mean() > min_val_loss) and (val.mean() < 1.5) and (epoch + 1) > num_epochs*0.1  :
                min_val_loss = val.mean()
                final_save_path = save_path + f'_final.pth'
                torch.save(model.state_dict(), final_save_path)
                print(f'Model saved at epoch {epoch + 1} to {final_save_path}')


    train_losses_np = [loss.numpy() for loss in train_losses]
    val_losses_np = [loss.numpy() for loss in val_losses]


    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting Training and Validation Losses
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses_np, label='Training Loss', color='tab:blue')
    ax1.plot(val_losses_np, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Create another y-axis for R2 Scores
    if not classification:
        ax2 = ax1.twinx()
        ax2.set_ylabel('R2 Score', color='tab:green')

        # Assuming val_r2_scores is a list of lists (one list per epoch)
        for i in range(len(val_r2_scores[0])):  # Loop over each target dimension
            r2_scores = [scores[i] for scores in val_r2_scores]
            ax2.plot(r2_scores, label=f'R2 Score y{i}', linestyle='--')

        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')

    else :
        ax2 = ax1.twinx()
        ax2.set_ylabel('F1 Score', color='tab:green')

        # Assuming val_r2_scores is a list of lists (one list per epoch)


        ax2.plot(val_f1_scores, label=f'f1 Score ', linestyle='--')

        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')

    # Set title and layout
    plt.title('Training and Validation Metrics per Epoch')
    fig.tight_layout()  # To prevent overlapping

    # Show the plot
    plt.show(block=False)



    if save_path:
        return train_losses, val_losses, val_r2_scores , final_save_path
    else :
        return train_losses, val_losses, val_r2_scores
###############################################################################

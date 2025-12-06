import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from tqdm import tqdm
import time
from DataLoading import create_dataloaders
from model import createModel
def _get_gradient_norm(model):
    """Helper to get gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return f"{total_norm:.2f}"

def train_model(model, dataloaders, epochs=50, lr=5e-4,  
                patience=5, clip_grad=1.0, device=None, 
                weight_decay=1e-5, pos_weight=3.004):     
    """
    Train model according to paper specifications.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float).to(device)
        print(f"Using class weighting: pos_weight={pos_weight:.1f}")
        loss_fn = nn.BCELoss(weight=pos_weight_tensor, reduction='mean')
    else:
        print("No class weighting applied")
        loss_fn = model.loss_fn
 
    optimizer = optim.Adam(model.parameters(), 
                          lr=lr, 
                          betas=(0.9, 0.999),
                          weight_decay=weight_decay)
    
    print(f"Using L2 regularization: weight_decay={weight_decay}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )

    history = {
        'train_loss': [], 'train_auc': [], 'train_logloss': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_logloss': [], 'val_acc': [],
        'best_val_auc': 0.0, 'epochs_no_improve': 0, 'best_epoch': 0,
        'learning_rates': []
    }
### stats 
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Train samples: {len(dataloaders['train'].dataset):,}")
    print(f"  Val samples: {len(dataloaders['val'].dataset):,}")
    print(f"  Batch size: {dataloaders['train'].batch_size}")
    print(f"  Learning rate: {lr} (paper: 5e-4)")
    print(f"  Gradient clipping: {clip_grad}")
    print(f"  Weight decay (L2): {weight_decay}")
    print(f"  Early stopping patience: {patience} (paper: 5)")

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        start_time = time.time()

        model.train()
        train_loss = 0.0
        y_true_train, y_pred_train, y_prob_train = [], [], []

        train_bar = tqdm(dataloaders['train'], desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(train_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            y_hat, _ = model(batch, compute_loss=False)  # Get predictions
            
            labels = batch['label']
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            loss = loss_fn(y_hat, labels)
            optimizer.zero_grad()
            loss.backward()
            
                #### should recheck this part, later ill chack it.
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            train_loss += loss.item() * batch['label'].size(0)
            y_true_train.extend(batch['label'].cpu().numpy().flatten())
            y_prob_train.extend(y_hat.detach().cpu().numpy().flatten())
            y_pred_train.extend((y_hat.detach().cpu().numpy() > 0.5).astype(int).flatten())

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                'grad_norm': _get_gradient_norm(model)})

        train_loss /= len(dataloaders['train'].dataset)
        train_auc = roc_auc_score(y_true_train, y_prob_train)
        train_logloss = log_loss(y_true_train, y_prob_train)
        train_acc = accuracy_score(y_true_train, y_pred_train)

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_logloss'].append(train_logloss)
        history['train_acc'].append(train_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        print(f"\n TRAIN SUMMARY:")
        print(f"  Loss:     {train_loss:.6f}")
        print(f"  AUC:      {train_auc:.6f}")
        print(f"  LogLoss:  {train_logloss:.6f}")
        print(f"  Accuracy: {train_acc:.4f}")

        if 'val' in dataloaders:
            model.eval()
            val_loss = 0.0
            y_true_val, y_pred_val, y_prob_val = [], [], []

            with torch.no_grad():
                val_bar = tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1}")
                for batch in val_bar:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    y_hat, _ = model(batch, compute_loss=False)
                    
                    labels = batch['label']
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(-1)
                    loss = loss_fn(y_hat, labels)
                    val_loss += loss.item() * batch['label'].size(0)

                    y_true_val.extend(batch['label'].cpu().numpy().flatten())
                    y_prob_val.extend(y_hat.cpu().numpy().flatten())
                    y_pred_val.extend((y_hat.cpu().numpy() > 0.5).astype(int).flatten())

            val_loss /= len(dataloaders['val'].dataset)
            val_auc = roc_auc_score(y_true_val, y_prob_val)
            val_logloss = log_loss(y_true_val, y_prob_val)
            val_acc = accuracy_score(y_true_val, y_pred_val)

            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_logloss'].append(val_logloss)
            history['val_acc'].append(val_acc)

            print(f"\n VALIDATION SUMMARY:")
            print(f"  Loss:     {val_loss:.6f}")
            print(f"  AUC:      {val_auc:.6f}")
            print(f"  LogLoss:  {val_logloss:.6f}")
            print(f"  Accuracy: {val_acc:.4f}")
            
            auc_gap = train_auc - val_auc
            loss_gap = val_loss - train_loss
            print(f"  Gap - AUC: {auc_gap:.4f}, Loss: {loss_gap:.4f}")

            scheduler.step(val_auc)
            
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and current_lr < history['learning_rates'][-2]:
                print(f"   Learning rate reduced to: {current_lr:.6f}")
################# early stopping 
            if val_auc > history['best_val_auc']:
                history['best_val_auc'] = val_auc
                history['best_epoch'] = epoch + 1
                history['epochs_no_improve'] = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'train_auc': train_auc,
                    'train_loss': train_loss,
                }, 'best_model.pth')
                print(f"  ✓ Saved best model (AUC: {val_auc:.6f})")
            else:
                history['epochs_no_improve'] += 1
                print(f"   No improvement for {history['epochs_no_improve']}/{patience} epochs")

                if history['epochs_no_improve'] >= patience:
                    print(f"\n   Early stopping at epoch {epoch + 1}")
                    print(f"  Best validation AUC: {history['best_val_auc']:.6f} at epoch {history['best_epoch']}")
                    break

        epoch_time = time.time() - start_time
        print(f"\n Epoch time: {epoch_time:.2f}s")

    return history

def evaluate_model(model, dataloader, device=None):
    """Evaluate model with detailed metrics."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            y_hat, _ = model(batch, compute_loss=False)
            
            labels = batch['label']
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            loss = model.loss_fn(y_hat, labels)

            total_loss += loss.item() * batch['label'].size(0)
            all_predictions.extend(y_hat.cpu().numpy().flatten())
            all_labels.extend(batch['label'].cpu().numpy().flatten())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_predictions)
    logloss = log_loss(all_labels, all_predictions)
    
    binary_preds = (all_predictions > 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)

    return {
        'loss': avg_loss,
        'auc': auc,
        'logloss': logloss,
        'accuracy': acc,
        'predictions': all_predictions,
        'labels': all_labels
    }

if __name__ == "__main__":
    import sys
    
    sys.path.append('.')  

    TRAIN_PATH = "/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/train.parquet"
    VAL_PATH = "/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/valid.parquet"
    TEST_PATH = "/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/test.parquet"
    ITEM_INFO_PATH = "/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/item_info.parquet"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\nCreating DataLoaders...")
    dataloaders = create_dataloaders(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH,
        item_info_path=ITEM_INFO_PATH,
        batch_size=128, 
        max_seq_len=64,  
        num_workers=4,
        shuffle_train=True
    )

    print("\nCreating model...")
    train_dataset = dataloaders['train'].dataset
    model = createModel(train_dataset, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=15,           
        lr=5e-4,            
        patience=3,         
        clip_grad=5.0,      # idk what value to use, but 5 give good results
        device=device,
        weight_decay=1e-5,  
        pos_weight=3.004    
    )
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation AUC: {history['best_val_auc']:.6f}")
    print(f"Best epoch: {history['best_epoch']}")
    
 
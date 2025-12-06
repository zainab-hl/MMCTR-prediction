import torch
import polars as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from DataLoading import MMCTRDataset, MMCTRCollator
from model import create_complete_model_from_dataset 
from torch.utils.data import DataLoader
sys.path.append('.')

def prepare_test_data(test_path, item_info_path, max_seq_len=64):
    """Create test dataset without labels."""
    test_df = pl.read_parquet(test_path)
    test_dataset = MMCTRDataset(test_path, item_info_path, max_seq_len)
    
    return test_dataset, test_df

def load_trained_model(checkpoint_path, dataset, device=None, weight_only=False):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_complete_model_from_dataset(dataset, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation AUC: {checkpoint['val_auc']:.6f}")
    
    return model

def predict_test_set(model, test_dataset, batch_size=128, device=None):
    """Generate predictions for test set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    collator = MMCTRCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )
    
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            
            # Get predictions
            predictions = model.predict(batch)  # [B, 1]
            
            # Store
            all_predictions.extend(predictions.cpu().numpy().flatten())
            
            # Get IDs from batch if available
            if 'ID' in batch:
                all_ids.extend(batch['ID'].cpu().numpy())
    
    return np.array(all_predictions), np.array(all_ids)

### this function must be changed to be adapted to the required submission format .
def create_submission_file(test_df, predictions, output_path="submission.csv"):
    """Create submission file in required format."""
    submission_df = test_df.select(['ID']).to_pandas()
    submission_df['score'] = predictions
    
    submission_df = submission_df.sort_values('ID')
    
    submission_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    return submission_df

if __name__ == "__main__":
    # Paths - UPDATE THESE
    TEST_PATH = "test.parquet"
    ITEM_INFO_PATH = "item_info.parquet"
    CHECKPOINT_PATH = "best_model.pth"  ## the saved model after training
    SUBMISSION_PATH = "submission.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n1. Loading test data...")
    test_dataset, test_df = prepare_test_data(
        test_path=TEST_PATH,
        item_info_path=ITEM_INFO_PATH,
        max_seq_len=64  
    )
    
    print(f"Test samples: {len(test_dataset)}")
   
    print("\n2. Loading trained model...")
    model = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        dataset=test_dataset,  
        device=device
    )
    
    print("\n3. Making predictions...")
    predictions, ids = predict_test_set(
        model=model,
        test_dataset=test_dataset,
        batch_size=128,  # Same as training
        device=device
    )
   
    print("\n4. Creating submission file...")
    submission_df = create_submission_file(
        test_df=test_df,
        predictions=predictions,
        output_path=SUBMISSION_PATH
    )
#### some stats 
    print("\n5. Submission summary:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Mean prediction: {predictions.mean():.6f}")
    print(f"   Std prediction: {predictions.std():.6f}")
    print(f"   Min prediction: {predictions.min():.6f}")
    print(f"   Max prediction: {predictions.max():.6f}")
    print(f"\n   Prediction distribution:")
    print(f"   < 0.1: {(predictions < 0.1).sum()} samples")
    print(f"   0.1-0.5: {((predictions >= 0.1) & (predictions < 0.5)).sum()} samples")
    print(f"   0.5-0.9: {((predictions >= 0.5) & (predictions < 0.9)).sum()} samples")
    print(f"   >= 0.9: {(predictions >= 0.9).sum()} samples")
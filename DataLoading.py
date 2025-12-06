### dataloader
from logging import INFO
import numpy as np
import polars as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional

class MMCTRDataset(Dataset):
    """
    Dataset for WWW 2025 MM-CTR Challenge.
    Loads and preprocesses train/validation/test parquet files.
    """

    def __init__(self, data_path: str, item_info_path: str, max_seq_len: int = 64):
        self.max_seq_len = max_seq_len

        # Load data
        self.data_df = pl.read_parquet(data_path).to_pandas()
        self.item_info_df = pl.read_parquet(item_info_path).to_pandas()

        # Create mapping: item_id -> tags
        self.item_tags_dict = dict(zip(
            self.item_info_df['item_id'],
            self.item_info_df['item_tags']
        ))
        self.item_info_df = self.item_info_df.sort_values('item_id')
        emb_list = self.item_info_df['item_emb_d128'].tolist()
        self.frozen_embeddings = torch.tensor(emb_list, dtype=torch.float32)
       

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        history_ids = row['item_seq']
        history_ids = self._pad_or_truncate(history_ids, self.max_seq_len, pad_value=0)
        history_tags = [self._get_item_tags(item_id) for item_id in history_ids]

        target_id = row['item_id']
        target_tags = self._get_item_tags(target_id)

        # side features
        likes_level = row.get('likes_level', 0)
        views_level = row.get('views_level', 0)

        label = row.get('label', 0.0)

        sample = {
            'history_ids': torch.tensor(history_ids, dtype=torch.long),
            'history_tags': torch.tensor(history_tags, dtype=torch.long),
            'target_id': torch.tensor(target_id, dtype=torch.long),
            'target_tags': torch.tensor(target_tags, dtype=torch.long),
            'likes_level': torch.tensor(likes_level, dtype=torch.long),
            'views_level': torch.tensor(views_level, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }
        return sample

### helper functions
    def _get_item_tags(self, item_id: int) -> List[int]:
        if item_id == 0:
            return [0] * 5
        tags = self.item_tags_dict.get(item_id, [0, 0, 0, 0, 0])
        # Ensure exactly 5 tags
        if len(tags) < 5:
            tags = tags + [0] * (5 - len(tags))
        elif len(tags) > 5:
            tags = tags[:5]
        
        return tags

    def _pad_or_truncate(self, seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
        if not isinstance(seq, list):
            seq = list(seq)
        
        if len(seq) > max_len:
            return seq[-max_len:]
        elif len(seq) < max_len:
            return seq + [pad_value] * (max_len - len(seq))
        else:
            return seq

class MMCTRCollator:
    """
    Collator to batch samples together for DataLoader.
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batched = {}
        for key in batch[0].keys():
            tensors = [sample[key] for sample in batch]

            if key in ['history_ids', 'history_tags', 'target_tags']:
                batched[key] = torch.stack(tensors, dim=0)
            elif key == 'label':
                batched[key] = torch.stack(tensors, dim=0).unsqueeze(-1)
            else:
                batched[key] = torch.stack(tensors, dim=0)

        return batched


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    item_info_path: str = "item_info.parquet", ### later we'll change it to our file.
    batch_size: int = 128,
    max_seq_len: int = 64,
    num_workers: int = 4,
    shuffle_train: bool = True
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    dataloaders = {}
    collator = MMCTRCollator()

    # Train
    train_dataset = MMCTRDataset(train_path, item_info_path, max_seq_len)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # Validation
    if val_path:
        val_dataset = MMCTRDataset(val_path, item_info_path, max_seq_len)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )

    # Test
    if test_path:
        test_dataset = MMCTRDataset(test_path, item_info_path, max_seq_len)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )

    # Statistics
    print("\n" + "="*60)
    print("DataLoader Statistics")
    print("="*60)
    print(f"Train samples: {len(train_dataset)}")
    if val_path:
        print(f"Validation samples: {len(val_dataset)}")
    if test_path:
        print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")
    sample_batch = next(iter(dataloaders['train']))
    print("\nSample batch shapes:")
    for key, value in sample_batch.items():
        print(f"  {key}: {value.shape} ({value.dtype})")

    return dataloaders



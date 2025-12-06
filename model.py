### full model
import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tools import ItemEmbeddingLayer, SequentialFeatureLearning, SimpleSideFeatureEmbedding, DCNv2FeatureInteraction, CTRPredictionLayer, CTRLoss

class CompleteWWW2025Model(nn.Module):

    def __init__(
        self,
        # Item embedding parameters
        num_items,
        num_tags,
        frozen_multimodal_embeddings,  # Keep this name
        tag_count_per_item=5,
        item_id_emb_dim=16,  #### dim of embeddings (id, tags, views, likes,...) is a parameter, we can change it while experimenting
        tag_emb_dim=32,      
        multimodal_emb_dim=128,
        seq_num_layers=2,
        seq_num_heads=4,
        seq_dt=128,
        seq_k=16,
        seq_dropout=0.2,
        like_vocab_size=11,
        view_vocab_size=11,
        side_emb_dim=16,
        dcn_num_layers=3,   ### we can change the number of layers
        dcn_deep_hidden_dims=[1024, 512, 256],
        dcn_deep_output_dim=256,
        dcn_dropout=0.2,

        pred_hidden_dims=[64, 32],  ### better than 3 layers 128,64,32
        pred_dropout=0.0
    ):
        super().__init__()

        print("Initializing Complete WWW 2025 Model")
       
        self.item_embedding = ItemEmbeddingLayer(
            num_items=num_items,
            num_tag_ids=num_tags,
            frozen_mm=frozen_multimodal_embeddings,  
            tag_count_per_item=tag_count_per_item
        )
        self.item_emb_dim = item_id_emb_dim + tag_emb_dim * tag_count_per_item + multimodal_emb_dim
        print(f"Item embedding dimension: {self.item_emb_dim}")

        self.sequential_module = SequentialFeatureLearning(
            item_embedding_layer=self.item_embedding,
            embed_dim=self.item_emb_dim,
            dt=seq_dt,
            num_layers=seq_num_layers,
            num_heads=seq_num_heads,
            k=seq_k,
            dropout=seq_dropout
        )
        self.S_o_dim = seq_k * seq_dt + seq_dt
        print(f"Sequential output dimension (S_o): {self.S_o_dim}")

        self.side_embedding = SimpleSideFeatureEmbedding(
            like_vocab_size=like_vocab_size,
            view_vocab_size=view_vocab_size,
            emb_dim=side_emb_dim
        )
        self.side_dim = side_emb_dim * 2
        print(f"Side feature dimension: {self.side_dim}")


        dcn_input_dim = self.item_emb_dim + self.side_dim + self.S_o_dim
        print(f"DCNv2 input dimension: {dcn_input_dim}")

        self.feature_interaction = DCNv2FeatureInteraction(
            input_dim=dcn_input_dim,
            num_cross_layers=dcn_num_layers,
            deep_hidden_dims=dcn_deep_hidden_dims,
            deep_output_dim=dcn_deep_output_dim,
            dropout_rate=dcn_dropout
        )

        pred_input_dim = dcn_input_dim + dcn_deep_output_dim
        print(f"Prediction layer input dimension: {pred_input_dim}")

        self.prediction_layer = CTRPredictionLayer(
            input_dim=pred_input_dim,
            hidden_dims=pred_hidden_dims,
            dropout_rate=pred_dropout
        )

        self.loss_fn = CTRLoss(reduction='mean')
        
        print("="*60)
        print("Model initialization complete!")

    def forward(self, batch_data, compute_loss=False):
        """
        Forward pass through the complete model.
        
        Args:
            batch_data: dict containing:
                - history_ids: [B, N]
                - history_tags: [B, N, 5]
                - target_id: [B]
                - target_tags: [B, 5]
                - likes_level: [B]
                - views_level: [B]
                - label: [B, 1] (optional, for training)
            
            compute_loss: if True and label provided, compute loss
        
        Returns:
            y_hat: [B, 1] predicted CTR probability
            loss: scalar (if compute_loss=True and label provided)
        """
        expected_keys = {'history_ids', 'history_tags', 'target_id', 'target_tags', 
                        'likes_level', 'views_level'}
        
        missing_keys = expected_keys - set(batch_data.keys())
        if missing_keys:
            print(f"Warning: Missing keys in batch_data: {missing_keys}")
        
        target_emb = self.item_embedding(
            batch_data['target_id'], 
            batch_data['target_tags']
        )  # [B, item_emb_dim]   

        S_o = self.sequential_module(
            batch_data['history_ids'],
            batch_data['history_tags'],
            batch_data['target_id'],
            batch_data['target_tags']
        )  # [B, S_o_dim]

        e_side = self.side_embedding(
            batch_data['likes_level'],
            batch_data['views_level']
        )  # [B, side_dim]

        f_o = self.feature_interaction(target_emb, e_side, S_o)  # [B, dcn_input_dim + dcn_deep_output_dim]

        y_hat = self.prediction_layer(f_o)  # [B, 1] predicted CTR probability

        loss = None
        if compute_loss and 'label' in batch_data:
            label = batch_data['label']
            if label.dim() == 1:
                label = label.unsqueeze(-1)  # [B] -> [B, 1]
            
            if y_hat.shape != label.shape:
                print(f"Warning: Shape mismatch - y_hat: {y_hat.shape}, label: {label.shape}")
            
            loss = self.loss_fn(y_hat, label)

        return y_hat, loss

    def predict(self, batch_data):
        """Inference mode forward pass."""
        with torch.no_grad():
            y_hat, _ = self.forward(batch_data, compute_loss=False)
            return y_hat
#### helper function to create model from dataset
def createModel(dataset, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nCreating model on device: {device}")
    
    frozen_emb_tensor = dataset.frozen_embeddings.to(device)
    frozen_emb_tensor.requires_grad = False
    
    num_items = len(frozen_emb_tensor)
    
    all_tags = []
    for tags in dataset.item_tags_dict.values():
        all_tags.extend(tags)
    
    num_tags = max(all_tags) + 1 if all_tags else 1
    
    print(f"Dataset info: {num_items} items, {num_tags} unique tags")
    
    item_id_emb_dim = 16  
    tag_emb_dim = 32      
    multimodal_emb_dim = 128
    tag_count_per_item = 5
    
    actual_item_emb_dim = item_id_emb_dim + (tag_emb_dim * tag_count_per_item) + multimodal_emb_dim
    print(f"Actual item embedding dimension: {actual_item_emb_dim}")
    
    model = CompleteWWW2025Model(
        num_items=num_items,
        num_tags=num_tags,
        frozen_multimodal_embeddings=frozen_emb_tensor,
        tag_count_per_item=tag_count_per_item,
        item_id_emb_dim=item_id_emb_dim,  # 16
        tag_emb_dim=tag_emb_dim,          # 32
        multimodal_emb_dim=multimodal_emb_dim,
        seq_num_layers=2,
        seq_num_heads=4,
        seq_dt=128,
        seq_k=16,
        seq_dropout=0.2,
        like_vocab_size=11,
        view_vocab_size=11,
        side_emb_dim=16,
        dcn_num_layers=3,
        dcn_deep_hidden_dims=[1024, 512, 256],
        dcn_deep_output_dim=256,
        dcn_dropout=0.2,
        pred_hidden_dims=[64, 32],
        pred_dropout=0.0
    ).to(device)
    
    return model
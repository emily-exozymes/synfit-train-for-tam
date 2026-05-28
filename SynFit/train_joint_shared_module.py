import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from torch.nn.functional import gelu
import pandas as pd
import copy 
import torch.nn.functional as F
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
from sklearn.model_selection import KFold, train_test_split
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import argparse
from tqdm import tqdm
import torch.nn as nn
from transformers.models.esm.modeling_esm import EsmLayer, EsmModel, EsmLMHead, EsmEncoder, EsmEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from utils import spearman
import warnings
warnings.filterwarnings("ignore")



# Set random seed for reproducibility
def set_random_seed(seed_value=42):
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

# Argument parser
parser = argparse.ArgumentParser(description='Train 2-head joint model')
parser.add_argument('--protein', type=str, required=True, help='Protein name (e.g., BRCA1_HUMAN)')
parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use (e.g., "3,4,5,6")')
args = parser.parse_args()

def BT_loss(scores, golden_score):
    """Bradley-Terry loss function"""
    loss = torch.tensor(0., device=scores.device)
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss

def KLloss(logits, logits_reg):
    """KL divergence loss for regularization"""
    criterion_reg = torch.nn.KLDivLoss(reduction='mean', log_target=True)
    return criterion_reg(logits, logits_reg)

def compute_score(model, seq, mask, positions, wt_seq, mask_id, branch):
    device = seq.device
    
    # Mask the sequence for prediction
    masked_seq = seq.clone()
    batch_size = seq.size(0)
    
    for i in range(batch_size):
        if len(positions[i]) > 0:
            try:
                position = torch.tensor(list(map(int, positions[i].split(','))), device=device)
                # Add bounds checking
                valid_positions = position[(position >= 0) & (position < seq.size(1) - 2)]
                if len(valid_positions) > 0:
                    masked_seq[i, 1+valid_positions] = mask_id
            except (ValueError, IndexError) as e:
                print(f"Warning: Invalid position string '{positions[i]}' for sample {i}")
                continue
        else:
            # For wildtype, mask a safe position
            temp_position = min(75, seq.size(1) - 3)  # Ensure within bounds
            masked_seq[i, 1+temp_position] = mask_id

    # Compute the log probability of the masked token
    try:
        out = model(input_ids=masked_seq, attention_mask=mask, output_hidden_states=True, branch=branch)
        log_probs = torch.log_softmax(out, dim=-1)
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        print(f"masked_seq shape: {masked_seq.shape}, mask shape: {mask.shape}")
        raise e
    
    # Compute prediction scores
    scores = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        if len(positions[i]) > 0:
            try:
                position = torch.tensor(list(map(int, positions[i].split(','))), device=device)
                # Add bounds checking
                valid_positions = position[(position >= 0) & (position < seq.size(1) - 2)]
                if len(valid_positions) > 0:
                    log_prob = log_probs[i]
                    wt_i = wt_seq[i]
                    mt_i = seq[i]
                    scores[i] = torch.sum(log_prob[1+valid_positions, mt_i[1+valid_positions]] - 
                                        log_prob[1+valid_positions, wt_i[1+valid_positions]])
            except (ValueError, IndexError) as e:
                print(f"Warning: Error computing score for sample {i}: {e}")
                scores[i] = 0
        else:
            # For wildtype, score is 0
            scores[i] = 0
    
    return scores

# Modified LM Head
class EsmLMHead_Modified(EsmLMHead):
    """ESM Head for masked language modeling."""
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x

# Modified Encoder with shared layers
class EsmEncoder_Modified(nn.Module):
    def __init__(self, config):
        super(EsmEncoder_Modified, self).__init__()
        self.config = config
        
        # All 33 layers are shared
        self.shared_layers = nn.ModuleList([EsmLayer(config) for _ in range(33)])
        
        # Layer normalization after shared layers
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, 
                encoder_hidden_states=None, encoder_attention_mask=None, 
                past_key_values=None, use_cache=None, output_attentions=False, 
                output_hidden_states=False, return_dict=True, branch="A"):
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        
        # Pass through all 33 shared layers
        for i, layer_module in enumerate(self.shared_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states, attention_mask, layer_head_mask,
                encoder_hidden_states, encoder_attention_mask, 
                past_key_value, output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Apply layer normalization after shared layers
        hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, 
                                   all_self_attentions, all_cross_attentions] if v is not None)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states, past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states, attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

# Modified ESM Model
class EsmModel_Modified(EsmModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        
        # Replace the encoder with our modified version
        self.encoder = EsmEncoder_Modified(config)
        self.pooler = None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_values=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, branch="A"):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # FIXED: Pass attention_mask to embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids, 
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, 
            past_key_values_length=past_key_values_length,
        )
        
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict, branch=branch,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output, pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions,
        )

# Main model with 2 heads
class EsmForMaskedLM_2Head(EsmForMaskedLM):
    def __init__(self, config):
        # Don't call super().__init__ to avoid the parent's lm_head creation
        from transformers.modeling_utils import PreTrainedModel
        PreTrainedModel.__init__(self, config)
        
        self.config = config
        
        # Initialize modified ESM model and 2 LM heads
        self.esm = EsmModel_Modified(config)
        self.lm_head_a = EsmLMHead_Modified(config)  # Head 1
        self.lm_head_b = EsmLMHead_Modified(config)  # Head 2
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_output_embeddings(self):
        return self.lm_head_a.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head_a.decoder = new_embeddings

    def predict_contacts(self, tokens, attention_mask):
        return self.esm.predict_contacts(tokens, attention_mask=attention_mask)

    def forward(self, input_ids=None, attention_mask=None, branch=None, **kwargs):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            branch=branch,
            **kwargs
        )
        
        sequence_output = outputs[0]

        # Select the appropriate LM head based on branch
        if branch == "A":
            prediction_scores = self.lm_head_a(sequence_output)
        elif branch == "B": 
            prediction_scores = self.lm_head_b(sequence_output)
        else:
            prediction_scores = self.lm_head_a(sequence_output)  # Default to head A

        return prediction_scores

def load_model_weights(model_modified, baseline_models):
    """Load weights from 2 baseline models into the joint model"""
    if len(baseline_models) != 2:
        raise ValueError(f"Expected 2 baseline models, got {len(baseline_models)}")
    
    modified_params = dict(model_modified.named_parameters())
    
    # Add decoder weights to baseline params
    baseline_params = []
    for i, baseline in enumerate(baseline_models):
        params = dict(baseline.named_parameters())
        if 'lm_head.decoder.weight' not in params:
            params['lm_head.decoder.weight'] = baseline.state_dict()['lm_head.decoder.weight']
        baseline_params.append(params)
    
    # Load shared layers from first baseline (they should all be the same)
    for i in range(33):
        shared_layer_prefix = f"esm.encoder.shared_layers.{i}."
        baseline_layer_prefix = f"esm.encoder.layer.{i}."

        baseline_layers = {k: v for k, v in baseline_params[0].items() if k.startswith(baseline_layer_prefix)}

        for key, baseline_param in baseline_layers.items():
            modified_key = key.replace("layer", "shared_layers")
            if modified_key in modified_params:
                modified_params[modified_key].data.copy_(baseline_param.data)
    
    # Load LM heads from respective baselines
    head_mappings = [
        ('lm_head_a', 0),
        ('lm_head_b', 1)
    ]
    
    for head_name, baseline_idx in head_mappings:
        lm_head_components = ['bias', 'dense.weight', 'dense.bias', 'layer_norm.weight', 'layer_norm.bias', 'decoder.weight']
        
        for comp in lm_head_components:
            modified_key = f"{head_name}.{comp}"
            baseline_key = f"lm_head.{comp}"
            
            if baseline_key in baseline_params[baseline_idx] and modified_key in modified_params:
                modified_params[modified_key].data.copy_(baseline_params[baseline_idx][baseline_key].data)
                print(f"Loaded {baseline_key} from baseline {baseline_idx+1} into {modified_key}")

    # Load shared components (embeddings, layer norm)
    shared_components = {
        'esm.embeddings.word_embeddings.weight': baseline_params[0]['esm.embeddings.word_embeddings.weight'],
        'esm.embeddings.position_embeddings.weight': baseline_params[0]['esm.embeddings.position_embeddings.weight'],
        'esm.encoder.emb_layer_norm_after.weight': baseline_params[0]['esm.encoder.emb_layer_norm_after.weight'],
        'esm.encoder.emb_layer_norm_after.bias': baseline_params[0]['esm.encoder.emb_layer_norm_after.bias']
    }
    
    for comp_key, baseline_param in shared_components.items():
        if comp_key in modified_params:
            modified_params[comp_key].data.copy_(baseline_param.data)
            print(f"Loaded {comp_key} from baseline")

    print("2-head model weights loading completed!")

def freeze_lm_heads(model_modified):
    """Freeze all LM heads"""
    for head_name in ['lm_head_a', 'lm_head_b']:
        for name, param in model_modified.named_parameters():
            if name.startswith(head_name):
                param.requires_grad = False
                print(f"Freezing {name}")
    
    print("All LM heads have been frozen.")

def train(model, device, dataloader, optimizer, epoch, logits_reg, tokenizer, primary_device):
    model.train()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, LR: {current_lr}')

    total_loss = 0.0
    gt1_list, gt2_list = [], []
    pred1_list, pred2_list = [], []

    for step, data in tqdm(enumerate(dataloader), desc=f'Training {epoch}'):
        try:
            # Unpack data directly (no wildtype concatenation needed)
            seq, mask, positions, wt_seq, gt1, gt2 = data

            # Move to primary device first
            seq = seq.to(primary_device)
            mask = mask.to(primary_device)
            wt_seq = wt_seq.to(primary_device)
            gt1, gt2 = gt1.to(primary_device), gt2.to(primary_device)
            
            # Compute scores for both heads
            scores1 = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id, branch='A')
            scores2 = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id, branch='B')
            
            # Compute losses
            loss1 = BT_loss(scores1, gt1)
            loss2 = BT_loss(scores2, gt2)

            # Regularization for each head (use a single sample to avoid memory issues)
            reg_losses = []
            wt_single = wt_seq[0:1]
            mask_single = mask[0:1]
            
            for branch in ['A', 'B']:
                out = model(input_ids=wt_single, attention_mask=mask_single, output_hidden_states=True, branch=branch)
                log_probs = torch.log_softmax(out, dim=-1)[0]
                # Move logits_reg to same device as log_probs
                logits_reg_device = logits_reg.to(log_probs.device)
                loss_reg = KLloss(log_probs, logits_reg_device)
                reg_losses.append(loss_reg)

            # Store predictions and ground truth
            pred1_list.extend(scores1.cpu().detach().numpy())
            pred2_list.extend(scores2.cpu().detach().numpy())
            
            gt1_list.extend(gt1.cpu().numpy())
            gt2_list.extend(gt2.cpu().numpy())

            # Combined loss with equal weighting (1/2 for each head)
            weight = 0.5
            loss = weight * (loss1 + reg_losses[0]) + weight * (loss2 + reg_losses[1])

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        except Exception as e:
            print(f"Error in training step {step}: {e}")
            print(f"Batch shapes - seq: {seq.shape if 'seq' in locals() else 'N/A'}, mask: {mask.shape if 'mask' in locals() else 'N/A'}")
            raise e

    # Calculate Spearman correlations
    total_loss /= len(dataloader)
    sp1 = spearman(np.array(gt1_list), np.array(pred1_list))
    sp2 = spearman(np.array(gt2_list), np.array(pred2_list))

    return total_loss, sp1, sp2

def evaluate(model, device, dataloader, epoch, tokenizer, primary_device, istest=False):
    model.eval()

    total_loss = 0.0
    gt1_list, gt2_list = [], []
    pred1_list, pred2_list = [], []

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), desc=f'Evaluation {epoch}'):
            try:
                # Unpack data directly (no wildtype concatenation needed)
                seq, mask, positions, wt_seq, gt1, gt2 = data

                seq = seq.to(primary_device)
                mask = mask.to(primary_device)
                wt_seq = wt_seq.to(primary_device)
                gt1, gt2 = gt1.to(primary_device), gt2.to(primary_device)

                # Compute scores for both heads
                scores1 = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id, branch='A')
                scores2 = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id, branch='B')
                
                # Compute losses
                loss1 = BT_loss(scores1, gt1)
                loss2 = BT_loss(scores2, gt2)

                # Combined loss with equal weighting
                loss = 0.5 * (loss1 + loss2)
                total_loss += loss.item()

                # Store predictions and ground truth
                pred1_list.extend(scores1.cpu().numpy())
                pred2_list.extend(scores2.cpu().numpy())
                
                gt1_list.extend(gt1.cpu().numpy())
                gt2_list.extend(gt2.cpu().numpy())
                
            except Exception as e:
                print(f"Error in evaluation step {step}: {e}")
                continue

    # Calculate Spearman correlations
    total_loss /= len(dataloader)
    sp1 = spearman(np.array(gt1_list), np.array(pred1_list))
    sp2 = spearman(np.array(gt2_list), np.array(pred2_list))
        
    return total_loss, sp1, sp2

def main():
    protein_name = args.protein
    fold = args.fold
    seed = args.seed
    
    # Set up devices
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Check GPU availability
    for gpu_id in gpu_ids:
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            print(f"GPU {gpu_id} not available")
            return
    
    set_random_seed(seed)
    
    print(f"🧬 2-Head Joint Training for {protein_name}")
    print(f"Fold: {fold}, Seed: {seed}")
    print(f"Using GPUs: {gpu_ids}, Primary device: {primary_device}")
    
    # Load tokenizer and dataset
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    
    # Import dataset from utils.py
    import yaml
    from utils import GeneralMultiFitnessDataset, GeneralMultiFitnessDataset_WildtypeOnly
    # Line ~170
    config_path = os.path.join(ROOT_DIR, "configs", "train_multi.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset = GeneralMultiFitnessDataset(tokenizer, protein_name)
    print(f"Dataset loaded with {len(dataset)} sequences for {dataset.get_num_metrics()} metrics")
    
    # Step 2: Split out test set first (e.g., 20% of the full data)
    full_indices = list(range(len(dataset)))
    trainval_indices, test_indices = train_test_split(
        full_indices, test_size=0.2, random_state=42, shuffle=True
    )

    batch_size = 1
    # Step 3: Now do KFold only on trainval_indices
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = args.fold
    k_folds = int(config['k_folds'])
    trainval_indices = np.array(trainval_indices)

    kfold = KFold(n_splits=k_folds, shuffle=True, 
                    random_state=0)
    for i, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        if fold == i:
            print(f'loading fold {fold}')
            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)
            train_size = int(float(config['train_ratio']) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], 
                                                        generator=torch.Generator().manual_seed(0))

    print(len(dataset))
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print("train index: " + str(train_idx[:20]))
    print("test index: " + str(test_indices[:20]))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Using batch size: {batch_size}")
    
    # Create joint model
    base_model = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D')
    config = copy.deepcopy(base_model.config)
    joint_model = EsmForMaskedLM_2Head(config)
    
    # Remove unused components (only if they exist)
    if hasattr(joint_model.esm, 'pooler') and joint_model.esm.pooler is not None:
        del joint_model.esm.pooler
    if hasattr(joint_model, 'lm_head'):
        del joint_model.lm_head
    if hasattr(joint_model.esm, 'contact_head'):
        del joint_model.esm.contact_head
    
    print("2-head joint model created successfully!")
    
    # Load baseline models (assuming they exist)
    baseline_models = []
    metric_names = dataset.get_metric_names()
    
    for i, metric_name in enumerate(metric_names):
        clean_name = metric_name[:-4] if metric_name.endswith('.csv') else metric_name
        baseline_path = os.path.join(ROOT_DIR, "results", protein_name, clean_name, f"fold_{fold}", f"seed_{seed}", "best_model.pth")
        
        if not os.path.exists(baseline_path):
            print(f"❌ Baseline model not found: {baseline_path}")
            return
        
        baseline_model = EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D')
        baseline_model.load_state_dict(torch.load(baseline_path, map_location='cpu'))
        baseline_models.append(baseline_model)
        print(f"✅ Loaded baseline {i+1}: {clean_name}")
    
    # Load weights into joint model
    load_model_weights(joint_model, baseline_models)
    
    # Freeze LM heads
    freeze_lm_heads(joint_model)
    
    # Move model to primary device first
    joint_model = joint_model.to(primary_device)
    
    # Setup for multi-GPU if specified
    if len(gpu_ids) > 1:
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        joint_model = torch.nn.DataParallel(joint_model, device_ids=gpu_ids)
    
    # Setup regularization logits using wildtype sequence
    wt_encoding = dataset.wt_encoding
    if wt_encoding.dim() == 1:
        wt_seq = wt_encoding.unsqueeze(0)  # Add batch dimension
    else:
        wt_seq = wt_encoding[0:1]  # Take first sequence if batched
    
    wt_mask = torch.ones_like(wt_seq)
    wt_seq, wt_mask = wt_seq.to(primary_device), wt_mask.to(primary_device)
    
    print(f"WT sequence shape for regularization: {wt_seq.shape}")
    
    # Move base model to device for regularization computation
    base_model = base_model.to(primary_device)
    
    with torch.no_grad():
        out = base_model(wt_seq, wt_mask, output_hidden_states=True).logits
        log_probs_reg = torch.log_softmax(out, dim=-1)[0].detach()
    
    # Move base model back to CPU to free GPU memory
    base_model = base_model.cpu()
    del base_model
    torch.cuda.empty_cache()
    
    # Setup optimizer with lower learning rate for stability
    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-6, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Setup results directory
    save_dir = os.path.join(ROOT_DIR, "joint_results", protein_name, f"fold_{fold}", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    result_file = os.path.join(save_dir, "results.txt")
    
    print(f"Results will be saved to: {save_dir}")
    
    # Training loop
    best_avg_spearman = -np.inf
    best_epoch = 0
    patience_counter = 0
    max_patience = 10
    
    print("Starting 2-head training...")
    
    for epoch in range(60):
        try:
            print(f"\n--- Epoch {epoch + 1}/60 ---")
            
            loss_train, sp1_train, sp2_train = train(
                joint_model, primary_device, train_loader, optimizer, epoch, log_probs_reg, tokenizer, primary_device)
            
            loss_val, sp1_val, sp2_val = evaluate(
                joint_model, primary_device, val_loader, epoch, tokenizer, primary_device)
            
            loss_test, sp1_test, sp2_test = evaluate(
                joint_model, primary_device, test_loader, epoch, tokenizer, primary_device, istest=True)

            # Print results
            print(f"Train - Loss: {loss_train:.4f}, Spearman: [{sp1_train:.3f}, {sp2_train:.3f}]")
            print(f"Val   - Loss: {loss_val:.4f}, Spearman: [{sp1_val:.3f}, {sp2_val:.3f}]")
            print(f"Test  - Loss: {loss_test:.4f}, Spearman: [{sp1_test:.3f}, {sp2_test:.3f}]")

            # Save results to file
            with open(result_file, "a") as file:
                file.write(f"Epoch {epoch}: Spearman1 train: {sp1_train}, Spearman2 train: {sp2_train}\n")
                file.write(f"Epoch {epoch}: Spearman1 validation: {sp1_val}, Spearman2 validation: {sp2_val}\n")
                file.write(f"Epoch {epoch}: Spearman1 Test: {sp1_test}, Spearman2 Test: {sp2_test}\n")
                file.write(f"Epoch {epoch}: Training loss: {loss_train}\n")
                file.write(f"Epoch {epoch}: Validation loss: {loss_val}\n")
                file.write(f"Epoch {epoch}: Test loss: {loss_test}\n")
                file.write(f"\n")
            
            # Track best model based on average validation Spearman
            avg_val_spearman = np.mean([sp1_val, sp2_val])
            if avg_val_spearman > best_avg_spearman:
                best_avg_spearman = avg_val_spearman
                best_epoch = epoch
                patience_counter = 0
                print(f"✅ New best model at epoch {epoch}: avg Spearman = {avg_val_spearman:.4f}")
                
                # Save best model
                best_model_path = os.path.join(save_dir, "best_model.pth")
                try:
                    if hasattr(joint_model, 'module'):  # If DataParallel
                        torch.save(joint_model.module.state_dict(), best_model_path)
                    else:
                        torch.save(joint_model.state_dict(), best_model_path)
                    print(f"Model saved to {best_model_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
            else:
                patience_counter += 1
            
            scheduler.step()
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {max_patience} epochs without improvement!")
                break
                best_model
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            print("Continuing to next epoch...")
            continue
    
    print(f"\n🎉 2-Head joint training completed!")
    print(f"Best avg validation Spearman: {best_avg_spearman:.4f} at epoch {best_epoch}")
    print(f"Metric mapping:")
    for i, metric_name in enumerate(metric_names):
        print(f"  Spearman{i+1}: {metric_name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
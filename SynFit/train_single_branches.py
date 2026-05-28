import sys 
import os 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import torch.nn as nn 
import copy 
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
from sklearn.model_selection import KFold
import os
import argparse
import yaml
from tqdm import tqdm

from utils import GeneralMultiFitnessDataset, spearman

import random
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# torch.hub.set_dir('/work/kerr')
torch.set_num_threads(1)

from torch.utils.data import Subset

import warnings

parser = argparse.ArgumentParser(description='Train baseline models for multi-fitness proteins')
parser.add_argument('--protein', type=str, required=True, help='Protein name (e.g., BLAT_ECOLX)')
parser.add_argument('--baseline_idx', type=int, required=True, help='Which baseline to train (0-based index)')
parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold (0-4)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
args = parser.parse_args()

def set_random_seed(seed_value=0):
    # Set the seed for the random number generator for Python's random library
    random.seed(seed_value)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed_value)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed_value)

    # If using GPUs, set the seed for all GPU operations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

set_random_seed(seed_value=args.seed)

def BT_loss(scores, golden_score):
    """Bradley-Terry loss function"""
    loss = torch.tensor(0.)
    loss = loss.to(scores.device)
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    
    if torch.isnan(loss):
        print(f"NaN BT loss detected! scores: {scores[:5]}, golden_score: {golden_score[:5]}")
    return loss

def KLloss(logits, logits_reg):
    """KL divergence loss for regularization"""
    criterion_reg = torch.nn.KLDivLoss(reduction='mean', log_target=True)
    loss = criterion_reg(logits, logits_reg)
    
    if torch.isnan(loss):
        print(f"NaN KL loss detected!")
    return loss

def compute_score(model, seq, mask, positions, wt_seq, mask_id):
    device = seq.device
    
    ''' Mask the sequence for prediction '''
    masked_seq = seq.clone()
    batch_size = seq.size(0)
    
    for i in range(batch_size):
        if len(positions[i]) > 0:
            position = torch.tensor(list(map(int, positions[i].split(','))))
            masked_seq[i, 1+position] = mask_id
        elif len(positions[i]) == 0:
            # For wildtype, mask a random position
            temp_position = 75
            masked_seq[i, 1+temp_position] = mask_id

    ''' Compute the log probability of the masked token '''
    out = model(masked_seq, mask, output_hidden_states=True).logits
    log_probs = torch.log_softmax(out, dim=-1)
    
    ''' Compute prediction scores '''
    scores = torch.zeros(batch_size).to(device)
    for i in range(batch_size):
        if len(positions[i]) > 0:
            position = torch.tensor(list(map(int, positions[i].split(','))))
            log_prob = log_probs[i]
            wt_i = wt_seq[i]
            mt_i = seq[i]
            score = torch.sum(log_prob[1+position, mt_i[1+position]] - log_prob[1+position, wt_i[1+position]])
            scores[i] = score
        elif len(positions[i]) == 0: 
            # For wildtype, score is 0
            manual_wt_score = 0
            scores[i] = manual_wt_score
    
    # Check for NaN scores
    if torch.isnan(scores).any():
        print(f"NaN scores detected in compute_score!")
        print(f"positions: {positions[:3]}")
        print(f"scores: {scores}")
    
    return scores

def train(model, device, dataloader, optimizer, epoch, tokenizer, logits_reg, baseline_idx):
    model.train()

    gt_list, pred_list = [], []

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, LR: {current_lr}')

    total_loss = 0.0

    for step, data in tqdm(enumerate(dataloader), desc=f'Training {epoch}'):
        try:
            # Unpack data
            seq, mask, positions, wt_seq = data[:4]
            metrics = data[4:]  # All the ground truth metrics
            
            seq = seq.to(device)
            mask = mask.to(device)
            wt_seq = wt_seq.to(device)
            
            # Get the specific metric for this baseline
            gt = metrics[baseline_idx].to(device)
            
            # Check for NaN in ground truth
            if torch.isnan(gt).any():
                print(f"NaN detected in ground truth at step {step}!")
                continue
            
            scores = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id)
            
            # Check scores before loss calculation
            if torch.isnan(scores).any():
                print(f"NaN scores detected at step {step}!")
                continue
                
            loss = BT_loss(scores, gt)
            
            if torch.isnan(loss):
                print(f"NaN loss at step {step}, skipping...")
                continue

            # Regularization using wildtype sequence
            wt_seq_single = wt_seq[0:1]
            wt_mask_single = mask[0:1]
            out = model(wt_seq_single, wt_mask_single, output_hidden_states=True).logits
            log_probs = torch.log_softmax(out, dim=-1)[0]
            loss_reg = KLloss(log_probs, logits_reg)
            
            if torch.isnan(loss_reg):
                print(f"NaN regularization loss at step {step}, using only main loss...")
                total_loss_combined = loss
            else:
                total_loss_combined = loss + loss_reg

            if torch.isnan(total_loss_combined):
                print(f"Total loss is NaN at step {step}, skipping...")
                continue

            scores_snapshot = scores.clone().detach().cpu().numpy()
            pred_list.extend(scores_snapshot)
            
            gt_list.extend(gt.cpu().numpy())

            optimizer.zero_grad()
            total_loss_combined.backward()
            
            # Check gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if total_norm > 1000:  # Gradient clipping
                print(f"Large gradient norm: {total_norm}, clipping...")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += total_loss_combined.item()
            
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue

    if len(pred_list) == 0 or len(gt_list) == 0:
        print("No valid predictions collected!")
        return float('inf'), 0.0
        
    gt_list = np.array(gt_list)
    pred_list = np.array(pred_list)
    sp = spearman(gt_list, pred_list)
    total_loss /= len(dataloader)
    
    print(f"Training evaluation scores shape: {len(pred_list)}")
    print(f"Training loss: {total_loss}, Spearman: {sp}")

    return total_loss, sp

def evaluate(model, device, dataloader, epoch, tokenizer, baseline_idx, istest=False):
    model.eval()

    total_loss = 0.0
    gt_list, pred_list = [], []

    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), desc=f'Evaluation {epoch}'):
            try:
                # Unpack data
                seq, mask, positions, wt_seq = data[:4]
                metrics = data[4:]  # All the ground truth metrics
                
                seq = seq.to(device)
                mask = mask.to(device)
                wt_seq = wt_seq.to(device)
                
                # Get the specific metric for this baseline
                gt = metrics[baseline_idx].to(device)
                
                if torch.isnan(gt).any():
                    continue

                scores = compute_score(model, seq, mask, positions, wt_seq, mask_id=tokenizer.mask_token_id)
                
                if torch.isnan(scores).any():
                    continue
                    
                loss = BT_loss(scores, gt)
                
                if torch.isnan(loss):
                    continue

                total_loss += loss.item()

                gt_list.extend(gt.cpu().numpy())
                pred_list.extend(scores.cpu().numpy())
                
            except Exception as e:
                print(f"Error in evaluation step {step}: {e}")
                continue

    if len(pred_list) == 0:
        return float('inf'), 0.0
        
    total_loss /= len(dataloader)
    gt_list = np.array(gt_list)
    pred_list = np.array(pred_list)
    
    print(f"Evaluation scores shape: {len(pred_list)}")
    sp = spearman(gt_list, pred_list)
        
    return total_loss, sp

def freeze_selected_layers(basemodel, layers):
    """
    Freeze all layers in the basemodel except the last layers in the encoder,
    the contact head, the lm head, and the layer normalization after the encoder layers.
    """
    # Freeze all parameters initially
    for name, param in basemodel.named_parameters():
        param.requires_grad = False
        
    for i in range(layers, 33):
        for name, param in basemodel.esm.encoder.layer[i].named_parameters():
            param.requires_grad = True
            print(f"Unfrozen layer {name} in basemodel encoder layer {i}")

    # Unfreeze the lm head
    for name, param in basemodel.lm_head.named_parameters():
        param.requires_grad = True
        print(f"Unfrozen layer {name} in lm head")

    print(f"All layers frozen except the last {33-layers} encoder layers, contact head, lm head, and layer norm after the encoder.")
    
def sanity_check_requires_grad(model):
    """
    Prints the requires_grad status of each parameter in the model.
    """
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

def main():
    # Set random seed
    set_random_seed(args.seed)
    
    # Load tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Remove weight tying
    model.lm_head.decoder.weight = nn.Parameter(model.esm.embeddings.word_embeddings.weight.clone())
    assert not model.lm_head.decoder.weight is model.esm.embeddings.word_embeddings.weight
    
    model = model.to(args.device)
    freeze_selected_layers(model, 33)  # Freeze first 20 layers
    sanity_check_requires_grad(model)
    
    # Load dataset
    dataset = GeneralMultiFitnessDataset(tokenizer, args.protein)
    
    # Check if baseline_idx is valid
    num_metrics = dataset.get_num_metrics()
    metric_names = dataset.get_metric_names()
    
    if args.baseline_idx >= num_metrics:
        raise ValueError(f"Baseline index {args.baseline_idx} is out of range. Protein {args.protein} has {num_metrics} metrics.")
    
    # Get the actual metric name for folder naming
    metric_name = metric_names[args.baseline_idx]
    # Remove .csv extension and use the full metric name
    if metric_name.endswith('.csv'):
        metric_name = metric_name[:-4]
    
    print(f"Training baseline {args.baseline_idx} for metric: {metric_name}")
    print(f"Total number of metrics: {num_metrics}")
    print(f"All metrics: {metric_names}")
    
    # Create save directory using actual metric name
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    save_dir = os.path.join(base_dir, args.protein, metric_name, f"fold_{args.fold}", f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(parent_dir, "configs", "train_multi.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Step 2: Split out test set first (e.g., 20% of the full data)
    # full_indices = list(range(len(dataset)))
    # trainval_indices, test_indices = train_test_split(
    #     full_indices, test_size=0.2, random_state=42, shuffle=True
    # )

    # Step 3: Now do KFold only on trainval_indices
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = args.fold
    k_folds = int(config['k_folds'])
    # trainval_indices = np.array(trainval_indices)

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

    # print(len(dataset))
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # print("train index: " + str(train_idx[:20]))
    # print("test index: " + str(test_indices[:20]))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Setup regularization logits using wildtype sequence
    wt_seq, wt_mask = dataset.wt_encoding, torch.ones_like(dataset.wt_encoding)
    wt_seq = wt_seq.to(args.device)
    wt_mask = wt_mask.to(args.device)
    out = model(wt_seq, wt_mask, output_hidden_states=True).logits
    log_probs_reg = torch.log_softmax(out, dim=-1)[0]
    log_probs_reg = log_probs_reg.detach()
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Training setup
    best_spearman = -np.inf
    best_epoch = 0
    best_model_path = os.path.join(save_dir, f"best_model.pth")
    result_file = os.path.join(save_dir, f"results.txt")
    endure = 0 
    endure_time = 10
    
    # Training loop
    for epoch in range(80):
        loss_train, sp_train = train(model, args.device, train_loader, optimizer, epoch, tokenizer, log_probs_reg, args.baseline_idx)
        loss_val, sp_val = evaluate(model, args.device, val_loader, epoch, tokenizer, args.baseline_idx)
        loss_test, sp_test = evaluate(model, args.device, test_loader, epoch, tokenizer, args.baseline_idx, istest=True)

        with open(result_file, "a") as f:
            f.write(f"Epoch {epoch}: spearman training: {sp_train:.4f}\n")
            f.write(f"Epoch {epoch}: spearman validation: {sp_val:.4f}\n")
            f.write(f"Epoch {epoch}: spearman test: {sp_test:.4f}\n")
            f.write(f"Epoch {epoch}: Training loss: {loss_train:.4f}\n")
            f.write(f"Epoch {epoch}: Validation loss: {loss_val:.4f}\n")
            f.write(f"Epoch {epoch}: Test loss: {loss_test:.4f}\n\n")

        scheduler.step()

        if sp_val > best_spearman:
            best_spearman = sp_val
            best_epoch = epoch
            endure = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model at epoch {epoch}, validation spearman: {sp_val:.4f}")
        else: 
            endure += 1
            if endure >= endure_time:
                print("Early stop!")
                break

    print(f"Training completed. Best validation spearman: {best_spearman:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main() 
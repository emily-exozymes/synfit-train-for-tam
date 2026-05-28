import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import re

__all__ = ['GeneralMultiFitnessDataset', 'create_multi_fitness_dataloader']

def extract_wildtype_from_mutations(df):
    """
    Extract wildtype sequence by reverting single mutations.
    
    Args:
        df: DataFrame with 'mutant' and 'mutated_sequence' columns
    
    Returns:
        str: Wildtype sequence
    """
    # Find single mutations (pattern: letter + number + letter)
    single_mutation_pattern = r'^[A-Z]\d+[A-Z]$'
    single_mutations = df[df['mutant'].str.match(single_mutation_pattern)]
    
    if len(single_mutations) == 0:
        raise ValueError("No single mutations found to extract wildtype sequence")
    
    # Take the first single mutation
    first_mutation = single_mutations.iloc[0]
    mutant = first_mutation['mutant']
    mutated_seq = first_mutation['mutated_sequence']
    
    # Parse mutation (e.g., G234E -> original=G, position=234, new=E)
    original_aa = mutant[0]
    position = int(mutant[1:-1]) - 1  # Convert to 0-indexed
    new_aa = mutant[-1]
    
    # Verify the mutation is correct in the sequence
    if mutated_seq[position] != new_aa:
        raise ValueError(f"Mutation {mutant} doesn't match sequence at position {position}")
    
    # Revert the mutation to get wildtype
    wt_seq = list(mutated_seq)
    wt_seq[position] = original_aa
    wt_seq = ''.join(wt_seq)
    
    print(f"Extracted wildtype sequence from mutation {mutant}")
    print(f"Position {position+1}: {new_aa} -> {original_aa}")
    
    return wt_seq

class GeneralMultiFitnessDataset(Dataset):
    """
    General data loader for multi-fitness datasets from ProteinGym.
    Can handle any protein and any number of ground truth values.
    """
    def __init__(self, tokenizer, protein_name, data_dir='multi_fitness_data'):
        self.tokenizer = tokenizer
        self.protein_name = protein_name
        
        # Try to find the protein CSV file in multiple locations
        possible_csv_paths = [
            f"{protein_name}.csv",
            os.path.join(os.path.dirname(__file__), f"{protein_name}.csv"),
            os.path.join(os.getcwd(), f"{protein_name}.csv"),
            os.path.join(os.path.dirname(__file__), data_dir, f"{protein_name}.csv"),
            f"/work/tony/multi-main/src/nov_16_two_approaches/approach_1/cytc/ACCO_multi_exp/final_wet_lab_proteins_work/proteingym_benchmark_multi_fitness/{data_dir}/{protein_name}.csv"
        ]
        
        csv_file_found = False
        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_file = path
                csv_file_found = True
                print(f"Found {protein_name}.csv at: {path}")
                break
        
        if not csv_file_found:
            print(f"Tried to find {protein_name}.csv in: {possible_csv_paths}")
            raise FileNotFoundError(f"Protein CSV file {protein_name}.csv not found.")
        
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} mutants from {csv_file}")
        
        # Extract wildtype sequence from single mutations
        self.wt_seq = extract_wildtype_from_mutations(self.df)
        print(f'Protein: {protein_name}')
        print(f'wt_seq: {self.wt_seq}')
        print(f'wt_seq len: {len(self.wt_seq)}')
        
        # Get all DMS score columns (columns that start with 'DMS_score_')
        dms_score_cols = [col for col in self.df.columns if col.startswith('DMS_score_')]
        if not dms_score_cols:
            raise ValueError(f"No DMS score columns found in {csv_file}")
        
        print(f"Found {len(dms_score_cols)} DMS score columns: {dms_score_cols}")
        self.dms_score_cols = dms_score_cols
        
        # Store sequences and prepare data
        self.sequences = self.df['mutated_sequence'].values.tolist()
        self.seq_len = len(self.wt_seq)
        
        # Extract mutated positions from mutant column (e.g., "A126C" -> 126)
        self.df = self.df.copy()  # Make sure we can modify
        self.df['mutated_positions'] = self.df['mutant'].apply(self._extract_positions)
        
        # Filter out any NaN or invalid data
        print(f"Checking for invalid data...")
        initial_len = len(self.df)
        
        # Remove rows where any DMS score is NaN
        self.df = self.df.dropna(subset=dms_score_cols)
        
        # Remove rows with empty sequences
        self.df = self.df[self.df['mutated_sequence'].str.len() > 0]
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        if len(self.df) < initial_len:
            print(f"Filtered out {initial_len - len(self.df)} invalid rows, {len(self.df)} remaining")
        
        # Update sequences after filtering
        self.sequences = self.df['mutated_sequence'].values.tolist()
        
        # Calculate Spearman correlations between all pairs of DMS scores
        self.spearman_correlations = {}
        for i in range(len(dms_score_cols)):
            for j in range(i+1, len(dms_score_cols)):
                col1, col2 = dms_score_cols[i], dms_score_cols[j]
                # Remove rows with NaN values for correlation calculation
                valid_data = self.df[[col1, col2]].dropna()
                if len(valid_data) > 1:
                    corr, _ = spearmanr(valid_data[col1], valid_data[col2])
                    self.spearman_correlations[f"{col1}_vs_{col2}"] = corr
                    print(f'Spearman correlation between {col1} and {col2}: {corr:.4f}')
        
        # Tokenize protein sequences
        print(f"Tokenizing {len(self.sequences)} sequences...")
        self.encodings, self.attention_masks = self.tokenizer(
            self.sequences, 
            padding='max_length', 
            max_length=self.seq_len+2
        ).values()
        
        self.encodings = torch.tensor(self.encodings)
        self.attention_masks = torch.tensor(self.attention_masks)
        
        # Tokenize wildtype protein sequence
        wt_encoding, wt_attention_mask = self.tokenizer(
            [self.wt_seq], 
            padding='max_length', 
            max_length=self.seq_len+2
        ).values()
        self.wt_encoding = torch.tensor(wt_encoding)
        
        # Store all metrics (DMS scores) as tensors
        self.metrics = []
        for col in dms_score_cols:
            # Convert to tensor, should not have NaN values after filtering
            metric_values = self.df[col].values
            print(f"Metric {col}: min={metric_values.min():.3f}, max={metric_values.max():.3f}, mean={metric_values.mean():.3f}")
            metric_tensor = torch.tensor(metric_values, dtype=torch.float32)
            self.metrics.append(metric_tensor)
        
        print(f"Dataset initialized with {len(self.metrics)} fitness metrics")
        print(f"Final dataset size: {len(self.df)} sequences")
        
        # Debug: Print some sample data to verify alignment
        print(f"\nData alignment check:")
        for i in range(min(3, len(self.df))):
            print(f"Sample {i}: mutant={self.df['mutant'].iloc[i]}, positions={self.df['mutated_positions'].iloc[i]}, metrics=[{', '.join([f'{self.df[col].iloc[i]:.3f}' for col in dms_score_cols])}]")
    
    def _extract_positions(self, mutant):
        """Extract mutated positions from mutant string (e.g., 'A126C' -> '125')"""
        # Handle complex mutations by extracting all numbers
        positions = re.findall(r'\d+', mutant)
        if positions:
            # Convert to 0-indexed and return as comma-separated string
            zero_indexed_positions = [str(int(pos) - 1) for pos in positions]
            return ','.join(zero_indexed_positions)
        return ''
    
    def __getitem__(self, idx):
        """
        Returns: [encodings, attention_masks, mutated_positions, wt_encoding, metric1, metric2, ...]
        All elements are properly aligned and validated.
        """
        return [
            self.encodings[idx], 
            self.attention_masks[idx], 
            self.df['mutated_positions'].iloc[idx], 
            self.wt_encoding[0]
        ] + [metric[idx] for metric in self.metrics]
    
    def __len__(self):
        return len(self.df)
    
    def get_metric_names(self):
        """Get the names of all DMS score columns"""
        return self.dms_score_cols
    
    def get_spearman_correlations(self):
        """Get all Spearman correlations between metrics"""
        return self.spearman_correlations
    
    def get_num_metrics(self):
        """Get the number of fitness metrics"""
        return len(self.metrics)

class GeneralMultiFitnessDataset_WildtypeOnly(Dataset):
    """
    Dataset that returns only the wildtype sequence for regularization.
    """
    def __init__(self, tokenizer, protein_name, data_dir='multi_fitness_data'):
        self.tokenizer = tokenizer
        self.protein_name = protein_name
        
        # Get the number of metrics from the main dataset file
        possible_csv_paths = [
            f"{protein_name}.csv",
            os.path.join(os.path.dirname(__file__), f"{protein_name}.csv"),
            os.path.join(os.getcwd(), f"{protein_name}.csv"),
            os.path.join(os.path.dirname(__file__), data_dir, f"{protein_name}.csv"),
            f"/work/tony/multi-main/src/nov_16_two_approaches/approach_1/cytc/ACCO_multi_exp/final_wet_lab_proteins_work/proteingym_benchmark_multi_fitness/{data_dir}/{protein_name}.csv"
        ]
        
        csv_file_found = False
        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_file = path
                csv_file_found = True
                break
        
        if not csv_file_found:
            raise FileNotFoundError(f"Protein CSV file {protein_name}.csv not found.")
        
        df = pd.read_csv(csv_file)
        dms_score_cols = [col for col in df.columns if col.startswith('DMS_score_')]
        self.num_metrics = len(dms_score_cols)
        
        # Extract wildtype sequence
        self.wt_seq = extract_wildtype_from_mutations(df)
        self.seq_len = len(self.wt_seq)
        
        # Tokenize wildtype protein sequence
        wt_encoding, wt_attention_mask = self.tokenizer(
            [self.wt_seq], 
            padding='max_length', 
            max_length=self.seq_len+2
        ).values()
        self.wt_encoding = torch.tensor(wt_encoding)
        self.wt_attention_mask = torch.tensor(wt_attention_mask)
        
        # Create placeholder metrics (zeros for wildtype)
        self.wt_metrics = [torch.tensor([0.0]) for _ in range(self.num_metrics)]
    
    def __getitem__(self, idx):
        """
        Returns: [encodings, attention_masks, mutated_positions (empty), wt_encoding, metric1, metric2, ...]
        """
        return [self.wt_encoding[0], self.wt_attention_mask[0], '', self.wt_encoding[0]] + [metric[0] for metric in self.wt_metrics]
    
    def __len__(self):
        return 1

def create_multi_fitness_dataloader(tokenizer, protein_name, batch_size=32, shuffle=True, **kwargs):
    """
    Convenience function to create a DataLoader for a multi-fitness dataset.
    
    Args:
        tokenizer: The tokenizer to use
        protein_name: The name of the protein
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments for GeneralMultiFitnessDataset
    
    Returns:
        DataLoader: A PyTorch DataLoader for the multi-fitness dataset
    """
    from torch.utils.data import DataLoader
    
    dataset = GeneralMultiFitnessDataset(tokenizer, protein_name, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def spearman(x, y):
    """
    Compute Spearman correlation coefficient.
    """
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(x, y)
    return correlation

# Example usage:
# dataset = GeneralMultiFitnessDataset(tokenizer, "BLAT_ECOLX")
# dataloader = create_multi_fitness_dataloader(tokenizer, "BLAT_ECOLX", batch_size=32) 
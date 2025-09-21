import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def save_file(
    data: Any,
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Save data to various file formats with comprehensive error handling.
    
    Args:
        data: The data to be saved (DataFrame, dict, array, etc.)
        file_path: Path where the file should be saved
        file_type: Optional file type override ('pkl', 'json', 'npy', 'npz', 'csv', 'parquet')
        **kwargs: Additional format-specific parameters
    
    Returns:
        bool: True if successful, False if failed
    
    Examples:
        save_file(df, "data/output.csv")
        save_file(model, "models/model.pkl")
        save_file(tokenized_data, "data/tokenized.npz")
    """
    try:
        # Convert to Path object and ensure parent directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file type from extension if not provided
        if file_type is None:
            file_type = file_path.suffix.lower()[1:]  # Remove the dot
        
        # Save based on file type
        if file_type in ['pkl', 'pickle']:
            _save_pickle(data, file_path, **kwargs)
        
        elif file_type == 'json':
            _save_json(data, file_path, **kwargs)
        
        elif file_type == 'npy':
            _save_npy(data, file_path, **kwargs)
        
        elif file_type == 'npz':
            _save_npz(data, file_path, **kwargs)
        
        elif file_type in ['csv', 'tsv']:
            _save_csv(data, file_path, file_type, **kwargs)
        
        elif file_type in ['parquet', 'pq']:
            _save_parquet(data, file_path, **kwargs)
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}. "
                           f"Supported types: pkl, json, npy, npz, csv, tsv, parquet")
        
        logger.info(f"Successfully saved data to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {str(e)}")
        return False


def _save_pickle(data: Any, file_path: Path, **kwargs) -> None:
    """Save data using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, **kwargs)


def _save_json(data: Any, file_path: Path, **kwargs) -> None:
    """Save data as JSON."""
    if not isinstance(data, (dict, list)):
        raise ValueError("JSON data must be a dictionary or list")
    
    default_kwargs = {'indent': 2, 'ensure_ascii': False}
    default_kwargs.update(kwargs)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, **default_kwargs)


def _save_npy(data: Any, file_path: Path, **kwargs) -> None:
    """Save numpy array as .npy file."""
    if not isinstance(data, np.ndarray):
        raise ValueError("Data for .npy must be a numpy array")
    np.save(file_path, data, **kwargs)


def _save_npz(data: Any, file_path: Path, **kwargs) -> None:
    """Save data as .npz file (compressed numpy format)."""
    if isinstance(data, dict):
        np.savez_compressed(file_path, **data, **kwargs)
    elif isinstance(data, np.ndarray):
        np.savez_compressed(file_path, data=data, **kwargs)
    else:
        raise ValueError("NPZ data must be a dictionary of arrays or a single array")


def _save_csv(data: Any, file_path: Path, file_type: str, **kwargs) -> None:
    """Save data as CSV or TSV."""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("CSV/TSV data must be a pandas DataFrame or Series")
    
    sep = '\t' if file_type == 'tsv' else ','
    default_kwargs = {'index': False, 'sep': sep}
    default_kwargs.update(kwargs)
    
    data.to_csv(file_path, **default_kwargs)


def _save_parquet(data: Any, file_path: Path, **kwargs) -> None:
    """Save data as Parquet file."""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Parquet data must be a pandas DataFrame or Series")
    
    data.to_parquet(file_path, **kwargs)


# Additional specialized function for tokenized files
def save_tokenized_data(
    tokenized_data: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    base_name: str,
    **kwargs
) -> bool:
    """
    Specialized function for saving tokenized data with standard naming convention.
    
    Args:
        tokenized_data: Dictionary containing tokenized arrays ('input_ids', 'attention_mask', etc.)
        output_dir: Directory to save the files
        base_name: Base name for the files
        **kwargs: Additional parameters for save_file
    
    Returns:
        bool: True if all files saved successfully
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    for key, array in tokenized_data.items():
        file_path = output_dir / f"{base_name}_{key}.npy"
        if not save_file(array, file_path, **kwargs):
            success = False
    
    return success
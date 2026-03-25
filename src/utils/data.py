"""
GDA-VisionAssist - Data Utilities
Các hàm tiện ích cho data loading và saving.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration từ file YAML.
    
    Args:
        config_path: Đường dẫn file YAML
        
    Returns:
        Dict chứa config
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_results_json(results: Dict[str, Any], 
                       output_path: str,
                       append_timestamp: bool = True):
    """
    Lưu kết quả ra file JSON.
    
    Args:
        results: Dict kết quả
        output_path: Đường dẫn output
        append_timestamp: Thêm timestamp vào kết quả
    """
    if append_timestamp:
        results['timestamp'] = datetime.now().isoformat()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results_json(path: str) -> Dict[str, Any]:
    """
    Load kết quả từ file JSON.
    
    Args:
        path: Đường dẫn file JSON
        
    Returns:
        Dict chứa kết quả
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def get_checkpoint_info(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin từ checkpoint file.
    
    Args:
        checkpoint_path: Đường dẫn checkpoint
        
    Returns:
        Dict chứa thông tin (epoch, mIoU, keys, ...) hoặc None
    """
    import torch
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            "path": checkpoint_path,
            "size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
            "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["raw_state_dict"],
        }
        
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            if 'best' in checkpoint:
                info['best_metric'] = checkpoint['best']
            if 'best_miou' in checkpoint:
                info['best_miou'] = checkpoint['best_miou']
        
        return info
    except Exception as e:
        return {"path": checkpoint_path, "error": str(e)}

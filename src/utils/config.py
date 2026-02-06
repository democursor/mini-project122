"""Configuration management"""
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration loader with environment overrides"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv('APP_ENV', 'development')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        config_dir = Path('config')
        
        # Load default config
        default_path = config_dir / 'default.yaml'
        if default_path.exists():
            with open(default_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = self._get_default_config()
        
        # Load environment-specific config
        env_path = config_dir / f'{self.env}.yaml'
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
                config = self._deep_merge(config, env_config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'storage': {
                'pdf_directory': './data/pdfs',
                'max_file_size_mb': 50
            },
            'processing': {
                'max_concurrent_documents': 10,
                'retry_attempts': 3
            }
        }
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

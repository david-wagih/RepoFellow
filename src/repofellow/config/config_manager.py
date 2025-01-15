"""Configuration manager for handling model and agent configurations"""
from typing import Dict, Optional
from .agent_config import ModelConfig, AgentConfig, get_agent_config, get_model_config

class ConfigurationManager:
    """Manager for handling configurations"""
    
    def __init__(self):
        self._model_overrides: Dict[str, ModelConfig] = {}
        self._agent_overrides: Dict[str, AgentConfig] = {}
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration with possible override"""
        if model_name in self._model_overrides:
            return self._model_overrides[model_name]
        return get_model_config(model_name)
    
    def get_agent_config(self, agent_type: str) -> AgentConfig:
        """Get agent configuration with possible override"""
        if agent_type in self._agent_overrides:
            return self._agent_overrides[agent_type]
        return get_agent_config(agent_type)
    
    def override_model_config(self, model_name: str, config: ModelConfig) -> None:
        """Override model configuration temporarily"""
        self._model_overrides[model_name] = config
    
    def override_agent_config(self, agent_type: str, config: AgentConfig) -> None:
        """Override agent configuration temporarily"""
        self._agent_overrides[agent_type] = config
    
    def reset_overrides(self) -> None:
        """Reset all configuration overrides"""
        self._model_overrides.clear()
        self._agent_overrides.clear()

# Global configuration manager instance
config_manager = ConfigurationManager() 
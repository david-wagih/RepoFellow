"""Factory for creating language model instances"""
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ..config.agent_config import ModelConfig, ModelType

class ModelFactory:
    """Factory for creating language model instances"""
    
    @staticmethod
    def _create_openai_model(config: ModelConfig) -> BaseChatModel:
        """Create OpenAI model instance"""
        return ChatOpenAI(
            model=config.model_type.value,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
        )
    
    @staticmethod
    def _create_anthropic_model(config: ModelConfig) -> BaseChatModel:
        """Create Anthropic model instance"""
        return ChatAnthropic(
            model=config.model_type.value,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseChatModel:
        """Create a model instance based on configuration"""
        if config.model_type in [ModelType.GPT_4, ModelType.GPT_35_TURBO, ModelType.GPT_4_TURBO]:
            return cls._create_openai_model(config)
        elif config.model_type in [ModelType.CLAUDE_3_OPUS, ModelType.CLAUDE_3_SONNET]:
            return cls._create_anthropic_model(config)
        # Add more model providers as needed
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    @classmethod
    def create_model_with_fallback(
        cls,
        primary_config: ModelConfig,
        fallback_config: Optional[ModelConfig] = None,
    ) -> BaseChatModel:
        """Create a model with fallback option"""
        try:
            return cls.create_model(primary_config)
        except Exception as e:
            if fallback_config:
                try:
                    return cls.create_model(fallback_config)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both primary and fallback models failed. Primary: {str(e)}, Fallback: {str(fallback_error)}"
                    )
            raise RuntimeError(f"Model creation failed and no fallback provided: {str(e)}") 
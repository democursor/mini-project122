"""LLM client for RAG system with multiple provider support."""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class LLMClient:
    """Handles LLM API calls with fallback options."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ("openai", "google", or "ollama")
            model: Model name (will auto-detect for Google if not available)
            api_key: API key for provider (required for OpenAI and Google)
        """
        self.provider = provider.lower()
        self.requested_model = model
        self.model = model
        self.api_key = api_key
        self.client = self._initialize_client()
        logger.info(f"LLMClient initialized with provider={provider}, model={self.model}")
    
    def _list_google_models(self, genai) -> List[Dict]:
        """List all available Google Gemini models."""
        try:
            models = []
            for model in genai.list_models():
                # Check if model supports generateContent
                if 'generateContent' in model.supported_generation_methods:
                    models.append({
                        'name': model.name,
                        'display_name': model.display_name,
                        'description': model.description
                    })
            return models
        except Exception as e:
            logger.error(f"Error listing Google models: {e}")
            return []
    
    def _select_best_google_model(self, genai, requested_model: str) -> str:
        """
        Select the best available Google model with fallback logic.
        
        Priority order:
        1. Requested model (if available)
        2. gemini-1.0-pro
        3. gemini-pro
        4. First available model that supports generateContent
        """
        try:
            # List all available models
            available_models = self._list_google_models(genai)
            
            if not available_models:
                raise ValueError("No Google models available with generateContent support")
            
            logger.info(f"Found {len(available_models)} available Google models")
            
            # Extract model names (remove 'models/' prefix if present)
            model_names = [m['name'].replace('models/', '') for m in available_models]
            
            # Priority fallback list
            fallback_order = [
                requested_model,
                'gemini-1.0-pro',
                'gemini-pro',
                'gemini-1.5-pro',
                'gemini-1.5-flash'
            ]
            
            # Try each model in priority order
            for candidate in fallback_order:
                # Check both with and without 'models/' prefix
                if candidate in model_names:
                    logger.info(f"Selected Google model: {candidate}")
                    return candidate
                # Also check full name
                full_name = f"models/{candidate}"
                if full_name in [m['name'] for m in available_models]:
                    logger.info(f"Selected Google model: {candidate}")
                    return candidate
            
            # If no priority model found, use first available
            first_model = model_names[0]
            logger.warning(f"No priority models found, using first available: {first_model}")
            logger.info(f"Available models: {', '.join(model_names[:5])}")
            return first_model
            
        except Exception as e:
            logger.error(f"Error selecting Google model: {e}")
            raise
    
    def _initialize_client(self):
        """Initialize LLM client based on provider."""
        try:
            if self.provider == "openai":
                import openai
                if self.api_key:
                    return openai.OpenAI(api_key=self.api_key)
                else:
                    # Try to use environment variable
                    return openai.OpenAI()
            
            elif self.provider == "google":
                # Google AI Studio (Gemini)
                import google.generativeai as genai
                
                # Configure API key
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                else:
                    # Try to use environment variable GOOGLE_API_KEY
                    import os
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY not found in environment")
                    genai.configure(api_key=api_key)
                
                # Auto-detect and select best available model
                try:
                    self.model = self._select_best_google_model(genai, self.requested_model)
                    logger.info(f"Using Google model: {self.model}")
                except Exception as e:
                    logger.error(f"Failed to auto-detect Google model: {e}")
                    logger.warning(f"Falling back to requested model: {self.requested_model}")
                    # Keep the requested model and let it fail later with clear error
                
                return genai
                    
            elif self.provider == "ollama":
                # Local LLM via Ollama
                import requests
                return requests.Session()
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}", exc_info=True)
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """
        Generate response using configured LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated response text
        """
        try:
            logger.debug(f"Generating response with {self.provider}, prompt_length={len(prompt)}")
            
            if self.provider == "openai":
                return self._openai_generate(prompt, max_tokens, temperature)
            elif self.provider == "google":
                return self._google_generate(prompt, max_tokens, temperature)
            elif self.provider == "ollama":
                return self._ollama_generate(prompt, max_tokens, temperature)
                
        except Exception as e:
            # Enhanced error messages
            error_msg = str(e).lower()
            
            if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
                logger.error(f"❌ Authentication failed: Invalid API key for {self.provider}")
                raise ValueError(f"Authentication failed: Invalid API key for {self.provider}. Please check your API key.")
            
            elif "quota" in error_msg or "429" in error_msg:
                logger.error(f"❌ Quota exceeded for {self.provider}")
                raise ValueError(f"Quota exceeded for {self.provider}. Please check your account limits.")
            
            elif "404" in error_msg or "not found" in error_msg:
                logger.error(f"❌ Model not found: {self.model}")
                raise ValueError(f"Model '{self.model}' not found or not accessible. Please check available models.")
            
            else:
                logger.error(f"Error generating response: {e}", exc_info=True)
                raise
    
    def _openai_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].message.content
            logger.info(f"OpenAI response generated, length={len(generated_text)}")
            return generated_text
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
    
    def _google_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Google AI Studio (Gemini)."""
        try:
            # Try to create model - will fail if model not available
            try:
                model = self.client.GenerativeModel(self.model)
            except Exception as model_error:
                # If model creation fails, try to auto-detect again
                logger.warning(f"Model {self.model} not available, attempting auto-detection...")
                self.model = self._select_best_google_model(self.client, self.model)
                model = self.client.GenerativeModel(self.model)
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            generated_text = response.text
            logger.info(f"Google AI response generated, length={len(generated_text)}")
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide specific error messages
            if "404" in error_msg:
                logger.error(f"Google AI model error: {self.model} not found")
                # List available models for debugging
                try:
                    available = self._list_google_models(self.client)
                    model_names = [m['name'].replace('models/', '') for m in available]
                    logger.info(f"Available models: {', '.join(model_names[:5])}")
                except:
                    pass
            
            logger.error(f"Google AI API error: {e}", exc_info=True)
            raise
    
    def _ollama_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using local Ollama."""
        try:
            response = self.client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            generated_text = response.json()["response"]
            logger.info(f"Ollama response generated, length={len(generated_text)}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}", exc_info=True)
            raise

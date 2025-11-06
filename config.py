import logging
import os
import json
from typing import Optional
from dotenv import load_dotenv, set_key

# Добавлена константа для файла конфига
CONFIG_FILE = "config.json"

class ConfigManager:
    """
    Centralized configuration manager for app settings.
    Supports caching of .env loading and validation.
    """

    def __init__(self) -> None:
        self._env_loaded = False  # Флаг кеширования для dotenv

    def _ensure_env_loaded(self) -> None:
        """Ensures .env is loaded only once."""
        if not self._env_loaded:
            try:
                load_dotenv()
                self._env_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load .env file: {e}")

    def load_app_config(self) -> dict:
        """Loads application configuration from config.json."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading config.json: {e}. Returning empty config.")
                return {}
        return {}

    def save_app_config(self, config: dict) -> None:
        """Saves application configuration to config.json."""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except IOError as e:
            raise IOError(f"Failed to save config.json: {e}")

    def get_api_key(self, parent=None, prompt: bool = True) -> Optional[str]:
        """
        Retrieves the OpenAI API key from .env. If not found and prompt=True,
        prompts the user (via GUI if available, else console for fallback).
        """
        self._ensure_env_loaded()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key and prompt:
            try:
                # Попытка GUI-ввода
                from PyQt5.QtWidgets import QInputDialog, QLineEdit
                text, ok = QInputDialog.getText(parent, "Ввод API ключа",
                                                "Пожалуйста, введите ваш OpenAI API ключ:",
                                                QLineEdit.Password, "")
                if ok and text:
                    os.environ["OPENAI_API_KEY"] = text
                    set_key(".env", "OPENAI_API_KEY", text)
                    api_key = text
                else:
                    # Fallback: Консольный ввод если GUI недоступен
                    print("GUI is unavailable. Please enter your OpenAI API key:")
                    api_key = input().strip()
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
                        set_key(".env", "OPENAI_API_KEY", api_key)
            except ImportError:
                # GUI не доступен, использование консоли
                print("GUI libraries not available. Please enter your OpenAI API key:")
                api_key = input().strip()
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    set_key(".env", "OPENAI_API_KEY", api_key)

        if not api_key:
            logging.info("no api key")
            return None
        return api_key

    def get_model_name(self) -> str:
        """
        Retrieves the OpenAI model name from .env. Defaults to gpt-4o-mini.
        """
        self._ensure_env_loaded()
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def set_model_name(self, model_name: str) -> None:
        """
        Saves the OpenAI model name to .env.
        """
        set_key(".env", "OPENAI_MODEL", model_name)

    def get_pdf_directory(self) -> Optional[str]:
        """
        Retrieves the PDF directory from config.json.
        """
        config = self.load_app_config()
        return config.get("pdf_directory")

    def set_pdf_directory(self, directory_path: str) -> None:
        """
        Saves the PDF directory to config.json with validation.
        Raises ValueError if directory does not exist.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory '{directory_path}' does not exist or is not a directory.")
        config = self.load_app_config()
        config["pdf_directory"] = directory_path
        self.save_app_config(config)

# Глобальный экземпляр для использования в других модулях
config_manager = ConfigManager()

# Обратная совместимость: Перенаправление на методы класса
def load_app_config() -> dict:
    return config_manager.load_app_config()

def save_app_config(config: dict) -> None:
    config_manager.save_app_config(config)

def get_api_key(parent=None) -> Optional[str]:
    return config_manager.get_api_key(parent)

def get_model_name() -> str:
    return config_manager.get_model_name()

def set_model_name(model_name: str) -> None:
    config_manager.set_model_name(model_name)

def get_pdf_directory() -> Optional[str]:
    return config_manager.get_pdf_directory()

def set_pdf_directory(directory_path: str) -> None:
    config_manager.set_pdf_directory(directory_path)
from pathlib import Path

from utils import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):

    def __init__(self):
        self.openai_llm = 'gpt-4o'
        self.openai_embedding_model = 'text-embedding-3-small'
        self.resources_path = self.get_project_root() / "resources"
        self.pdf_file_path = self.resources_path / "64661631e57913001105970d.pdf"

    def configure(self, config_dict):
        for key, value in config_dict.items():
            self.__dict__[key] = value

    @staticmethod
    def get_project_root():
        return Path(__file__).parent.parent.parent.resolve()

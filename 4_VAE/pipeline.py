import os
import pickle
from abc import ABC, abstractmethod

class PipelineStep(ABC):
    """Базовый шаг пайплайна с возможностью кеширования результата."""

    def __init__(self, name, config, force=False):
        self.name = name
        self.config = config
        self.cache_path = os.path.join(config.CACHE_DIR, f"{name}.pkl")
        self.force = force

    def execute(self):
        """Проверяет кеш и либо загружает, либо запускает расчёт."""
        if not self.force and os.path.exists(self.cache_path):
            print(f"[{self.name}] Загрузка из кеша: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"[{self.name}] Выполнение...")
        result = self.run()

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"[{self.name}] Результат сохранён в {self.cache_path}")
        return result

    @abstractmethod
    def run(self):
        """Логика шага. Возвращает словарь с результатами."""
        pass
import os

class Config:
    # Пути
    OUTPUT_DIR = "output"
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

    # Данные
    VALIDATION_SPLIT = 0.1
    BATCH_SIZE = 256
    EPOCHS_AE = 10
    EPOCHS_VAE = 20

    # Аугментация
    AUGMENTATION = {
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.1
    }

    # Размерности латентного пространства для экспериментов
    LATENT_DIMS = [2, 8, 16]

    @classmethod
    def init_dirs(cls):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.FIGURES_DIR, exist_ok=True)
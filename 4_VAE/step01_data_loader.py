import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pipeline import PipelineStep

class DataLoaderStep(PipelineStep):
    def __init__(self, config, force=False):
        super().__init__("01_data_loader", config, force)

    def run(self):
        # 1. Загрузка
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 2. Нормализация и приведение формы к (N, 28, 28, 1)
        x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28, 28, 1)
        x_test  = (x_test.astype('float32') / 255.0).reshape(-1, 28, 28, 1)

        # 3. Аугментация (только для визуализации, не меняем тренировочный набор)
        datagen = ImageDataGenerator(**self.config.AUGMENTATION)
        datagen.fit(x_train)
        aug_iter = datagen.flow(x_train, y_train, batch_size=10, seed=42)
        aug_images, aug_labels = next(aug_iter)

        # 4. Визуализация оригиналов и аугментированных
        self._visualize(x_train[:10], aug_images)

        # 5. Статистика
        stats = {
            "train_shape": x_train.shape,
            "test_shape": x_test.shape,
            "min": float(x_train.min()),
            "max": float(x_train.max()),
            "mean": float(x_train.mean()),
            "std": float(x_train.std()),
            "class_distribution": dict(zip(*np.unique(y_train, return_counts=True)))
        }

        # 6. Сохраняем также сами данные как .npy (для быстрого доступа)
        np.save(self.config.CACHE_DIR + "/x_train.npy", x_train)
        np.save(self.config.CACHE_DIR + "/y_train.npy", y_train)
        np.save(self.config.CACHE_DIR + "/x_test.npy", x_test)
        np.save(self.config.CACHE_DIR + "/y_test.npy", y_test)

        result = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "stats": stats
        }
        return result

    def _visualize(self, originals, augmented):
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(augmented[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_title("Originals")
        axes[1, 0].set_title("Augmented")
        plt.tight_layout()
        fig_path = self.config.FIGURES_DIR + "/01_augmentation_examples.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Визуализация сохранена: {fig_path}")
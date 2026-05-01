from config import Config
from step01_data_loader import DataLoaderStep
from step02_ae import AEStep
from step03_vae import VAEStep
from step04_generate import GenerateStep   # <-- новая строка

def main():
    Config.init_dirs()

    loader = DataLoaderStep(Config, force=False)
    data = loader.execute()

    ae_step = AEStep(Config, force=False)
    ae_result = ae_step.execute()

    vae_step = VAEStep(Config, force=False)
    vae_result = vae_step.execute()

    # Шаг 4: Генерация данных
    gen_step = GenerateStep(Config, force=False)
    gen_step.execute()

    print("\nГенерация данных завершена успешно.")

if __name__ == "__main__":
    main()
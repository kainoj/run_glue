import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="defaults.yaml")
def main(config: DictConfig):

    from src.finetune import finetune
    from src.utils import print_config

    print_config(config, resolve=True)

    finetune(config)


if __name__ == "__main__":
    main()

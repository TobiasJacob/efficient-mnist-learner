import hydra
from hydra.core.config_store import ConfigStore

from eml.Config import Config
from eml.Datasets import load_data
from eml.Train import train

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config) -> None:
    data_loaders = load_data(cfg)
    train(cfg, *data_loaders)


if __name__ == "__main__":
    main()

import hydra

from eml.config import Config
from eml.train import train


@hydra.main(config_path=None)
def main(cfg: Config) -> None:
    print(cfg)
    cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()

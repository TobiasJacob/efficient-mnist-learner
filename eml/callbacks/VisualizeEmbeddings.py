import pytorch_lightning as pl
import torch
from tqdm import tqdm


class VisualizeEmbeddings(pl.Callback):
    """Pytorch-lightning callback to save the embeddings to Tensorboard"""

    def __init__(self, num_batches: int) -> None:
        """Create a new embedding visualizer.

        Args:
            num_batches (int): Number of batches used to save embeddings. A large number
            will generate more embeddings, however, tensorboard will be slower.
        """
        super().__init__()
        self.num_batches = num_batches

    def on_epoch_end(
        self, trainer: "pl.Trainer", auto_encoder: "pl.LightningModule"
    ) -> None:
        """Run at the end of epoch. Uses the auto_encoder to generate the embeddings.

        Args:
            trainer (pl.Trainer): The pytorch-ligthning trainer.
            auto_encoder (pl.LightningModule): The autoencoder.
        """
        embeddings = []
        all_imgs = []
        all_labels = []
        with torch.no_grad():
            batches = 0
            for dataloader in trainer.val_dataloaders:
                for imgs, labels in tqdm(
                    dataloader, "Generating embeddings", self.num_batches
                ):
                    embeddings.append(auto_encoder(imgs.to(auto_encoder.device)))
                    all_imgs.append(imgs)
                    all_labels.append(labels)
                    batches += 1
                    if batches > self.num_batches:
                        break
        embeddings = torch.cat(embeddings)
        all_imgs = torch.cat(all_imgs)
        all_labels = torch.cat(all_labels)

        trainer.logger.experiment.add_embedding(
            embeddings,
            metadata=all_labels,
            label_img=all_imgs,
            global_step=trainer.global_step,
        )

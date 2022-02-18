import pytorch_lightning as pl
import torch
from tqdm import tqdm


class VisualizeEmbeddings(pl.Callback):
    def __init__(self, num_batches: int) -> None:
        super().__init__()
        self.num_batches = num_batches

    def on_epoch_end(
        self, trainer: "pl.Trainer", auto_encoder: "pl.LightningModule"
    ) -> None:
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

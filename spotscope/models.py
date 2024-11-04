from torch import nn
import torch.nn.functional as F

from .modules import (
    ImageEncoder,
    ProjectionHead,
    ImageEncoder_BLEEP,
    ProjectionHead_BLEEP,
    ImageEncoder_STNET
)

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        image_embedding=1024,
        spot_embedding=4,
        projection_dim=128,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(image_embedding=image_embedding)
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding, projection_dim=projection_dim
        )  # aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(
            embedding_dim=spot_embedding, projection_dim=projection_dim
        )  # the number of celltype
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["annotations"]

        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)
        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T

        main_similarity = (images_similarity + spots_similarity) / 2
        targets = F.softmax(main_similarity / self.temperature, dim=-1)
        spots_loss = cross_entropy(logits, targets, reduction="none")
        images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


class BLEEP(nn.Module):
    def __init__(
        self,
        temperature=1,
        image_embedding=2048,
        spot_embedding=10,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_BLEEP()
        #         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead_BLEEP(
            projection_dim=256, embedding_dim=image_embedding
        )
        self.spot_projection = ProjectionHead_BLEEP(
            projection_dim=256, embedding_dim=spot_embedding
        )
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["annotations"]
        #         spot_features = self.spot_encoder(batch["reduced_expression"])

        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction="none")
        images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


class ST_NET(nn.Module):
    def __init__(
        self,
        image_embedding=1024,
        spot_embedding=6,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_STNET()
        #         self.spot_encoder = SpotEncoder()
        self.linear = nn.Linear(image_embedding, spot_embedding)
        

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["annotations"]

        logits = self.linear(image_features)
        loss = cross_entropy(logits, spot_features, reduction="none")
        return loss.mean()


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# if __name__ == "__main__":
#     images = torch.randn(8, 3, 224, 224)
#     input_ids = torch.randint(5, 300, size=(8, 25))
#     attention_mask = torch.ones(8, 25)
#     batch = {"image": images, "input_ids": input_ids, "attention_mask": attention_mask}

#     CLIP = CLIPModel()
#     loss = CLIP(batch)
#     print("")

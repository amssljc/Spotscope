from torch import nn
import timm


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self,
        model_name='resnet50',
        pretrained=True,
        trainable=True,
        features_only=True,
        image_embedding=1024
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            features_only=features_only,
            global_pool="avg",
        )
        self.image_embedding = image_embedding
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        # -3:512 -2:1024 -1:2048
        if self.image_embedding == 512:
            idx = -3
        elif self.image_embedding == 1024:
            idx = -2
        elif self.image_embedding == 2048:
            idx = -1
        return self.model(x)[idx].mean((-1, -2))

class ImageEncoder_BLEEP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_STNET(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='densenet121', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class ProjectionHead_BLEEP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
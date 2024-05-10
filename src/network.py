# Python packages
from termcolor import colored
from typing import Dict
from torchsummary import summary
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting

# class AlexNet(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

# [TODO: Optional] Rewrite this class if you want
# ResNet-18
class MyResNet(nn.Module):
    def __init__(self, num_classes=200):
        super(MyResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)

        # ResNet-18의 마지막 fully connected layer의 출력 특성을 num_classes로 변경
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

#ResNext
class MyResNeXt(nn.Module):
    def __init__(self, num_classes=200):
        super(MyResNeXt, self).__init__()
        self.resnext50 = models.resnext50_32x4d(pretrained=False)

        # ResNeXt의 마지막 fully connected layer의 출력 특성을 num_classes로 변경
        self.resnext50.fc = nn.Linear(self.resnext50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnext50(x)

# AlexNet
class MyAlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(MyAlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=False)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)

class MyAlexNet_BatchNorm(AlexNet):
    def __init__(self):
        super().__init__(num_classes=200)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),  # 첫 번째 Conv2d 뒤에 BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),  # 두 번째 Conv2d 뒤에 BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),  # 세 번째 Conv2d 뒤에 BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 네 번째 Conv2d 뒤에 BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 다섯 번째 Conv2d 뒤에 BatchNorm 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MyAlexNet_Deep(AlexNet):
    def __init__(self):
        super().__init__(num_classes=200)
        # 더 깊은 네트워크 구성
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyAlexNet':
            self.model = MyAlexNet()
        elif model_name == 'MyAlexNet_BatchNorm':
            self.model = MyAlexNet_BatchNorm()
        elif model_name == 'MyAlexNet_Deep':
            self.model = MyAlexNet_Deep()
        elif model_name == 'MyResNet':
            self.model = MyResNet(num_classes=num_classes)
        elif model_name == 'MyResNeXt':
            self.model = MyResNeXt(num_classes=num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1_score = MyF1Score(num_classes=num_classes)

        # Metric
        self.accuracy = MyAccuracy()

        # Hyperparameters
        self.save_hyperparameters()
        if torch.cuda.is_available():
            self.model.cuda()  # Ensure model is on GPU if available
        print(summary(self.model, (3, 224, 224)))  # Adjust input size as necessary

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # for cutmix
    # def training_step(self, batch, batch_idx):
    #     images, labels = batch  # collate_fn에서 반환된 이미지와 레이블
    #     outputs = self(images)  
    #     loss = torch.nn.functional.cross_entropy(outputs, labels)

    #     # 선택적: 정확도 계산
    #     _, preds = torch.max(outputs, dim=1)
    #     correct = preds.eq(labels.max(dim=1)[1]).sum().item()
    #     accuracy = correct / images.size(0)

    #     # 로깅
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)

    #     return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1_score(scores, y)
        self.log_dict({
                    'loss/test': loss,
                    'accuracy/val': accuracy,
                    'f1_score/val': f1.mean()
                }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])

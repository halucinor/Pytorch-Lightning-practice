# Pytorch-Lightning-practice

Pytorch-Lightning MNIST data classification train and inference with CNN

## Model

[source](mnist_cnn.ipynb)

```python
class CnnClassifier(LightningModule):
    def __init__(self, lr):
        super(CnnClassifier, self).__init__()
        self. lr = lr

        # 1 -> 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p = 0.5)
        )

        # 32 -> 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p = 0.5)
        )

        # 64 -> 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p = 0.5)
        )

        # 1152 -> 10
        self.fc = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x ,y = batch
        logits = self(x)
        acc = FM.accuracy(logits ,y)
        loss = F.cross_entropy(logits,y)
        metrics = {
            "val_acc" : acc,
            'val_loss' : loss
        }
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x ,y = batch
        logits = self(x)
        acc = FM.accuracy(logits ,y)
        loss = F.cross_entropy(logits,y)
        metrics = {
            "test_acc" : acc,
            'test_loss' : loss
        }
        self.log_dict(metrics)

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optim
```

## Result

![image](https://user-images.githubusercontent.com/10546369/174425313-f015c041-5b6f-4cc2-bfc7-0dfbeb539b84.png)
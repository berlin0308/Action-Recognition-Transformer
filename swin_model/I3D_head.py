import torch.nn as nn
# from mmcv.cnn import normal_init

class I3DHead(nn.Module):
    def __init__(self, num_classes, in_channels, **kwargs):
        super(I3DHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)

    def forward(self, x):
        # Reshape to [N, in_channels]
        x = x.reshape(x.size(0), -1)
        cls_score = self.fc_cls(x)  # [N, num_classes]
        return cls_score

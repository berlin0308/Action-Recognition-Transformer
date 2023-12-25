import torch.nn as nn
import torch
from swin_model.video_swin_transformer import SwinTransformer3D
from swin_model.I3D_head import I3DHead
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self, transformer_output_size, num_classes, freeze_layers=-1): # default: fine-tune all layers
        super(Classifier, self).__init__()
        self.transformer_output_size = transformer_output_size
        self.num_classes = num_classes

        # model = SwinTransformer3D(embed_dim=128, 
        #                   depths=[2, 2, 18, 2], 
        #                   num_heads=[4, 8, 16, 32], 
        #                   patch_size=(2,4,4), 
        #                   window_size=(16,7,7), 
        #                   drop_path_rate=0.4, 
        #                   patch_norm=True)
        model = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), # window: 16 or 8 
                          drop_path_rate=0.4, 
                          frozen_stages=freeze_layers, # -1: ft-all, 1~4: freeze 1~4 layers
                          patch_norm=True)
        
        checkpoint = torch.load('./swin_model/swin_base_patch244_window1677_sthv2.pth')
        # checkpoint = torch.load('./swin_model/swin_base_patch244_window1677_sthv2.pth')
       # checkpoint = torch.load('./swin_model/swin_base_patch244_window877_kinetics600_22k.pth')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 

        model.load_state_dict(new_state_dict) 
        self.video_transformer = model

        self.i3d_head = I3DHead(num_classes=self.num_classes, in_channels=transformer_output_size)

        self.print_trainable_parameters()
        

    def forward(self, x):

        transformer_output = self.video_transformer(x)
        # print(transformer_output.shape)
        i3d_output = self.i3d_head(transformer_output)
        return i3d_output

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, Size: {param.size()}")
            else:
                print(f"Non-trainable: {name}, Size: {param.size()}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import LULCEncoder,DSMEncoder
from torchvision.ops import DeformConv2d
from .dinov3 import LayerNorm2d,DINOv3,Decoder
 

class GeoConditionalAdaIN(nn.Module):
    def __init__(self, img_dim=64, geo_dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(img_dim)
        self.img_dim = img_dim
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, geo_dim * 2),
            nn.LayerNorm(geo_dim * 2),
            nn.GELU(),
        )

        self.geo_to_gamma = nn.Linear(geo_dim*2, img_dim)
        self.geo_to_beta  = nn.Linear(geo_dim*2, img_dim)

    def forward(self, instance_feat, geo_feat):
        B, C = instance_feat.shape  # [B, 64]
        feat_norm = self.norm(instance_feat)
        geo_feat = self.geo_proj(geo_feat)

        gamma = self.geo_to_gamma(geo_feat)  # [B,64]
        beta  = self.geo_to_beta(geo_feat)   # [B,64]

        fused_feat = feat_norm * gamma + beta  # [B,64]

        return fused_feat

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Conv2d(dim, 1, 1)
    def forward(self, x, mask):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.squeeze()
        weight = self.attn(x).sigmoid() * mask
        weight = weight / (weight.sum((2,3), keepdim=True) + 1e-6)
        return (x * weight).sum((2,3))


class CosGuidedMoEFusion(nn.Module):
    def __init__(self, dim, num_experts=2, dropout=0.0): 
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts


        self.experts = nn.ModuleList([
            nn.Conv2d(dim*2, dim, kernel_size=1) 
            for _ in range(num_experts)
        ])

        self.modal_gate = nn.Conv2d(dim*3, num_experts, kernel_size=1)
        self.norm = LayerNorm2d(dim)

    def forward(self, img, dsm, lulc, modality_mask):
        B, C, H, W = img.shape
        

        mask_spatial = modality_mask.unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]
        dsm = dsm * mask_spatial[:,1:2]  
        lulc = lulc * mask_spatial[:,2:3]


        ctx = dsm + lulc


        cos_sim = F.cosine_similarity(img.flatten(2), ctx.flatten(2), dim=1, eps=1e-8)
      
        img_guided = img * (1 + 0.1 * cos_sim.view(B,1,H,W))


        feat = torch.cat([img_guided, ctx], dim=1)
        

        gate = self.modal_gate(torch.cat([img, dsm, lulc], dim=1))
        gate = torch.softmax(gate, dim=1)  # [B, num_experts, H, W]
        fused = 0.0
        for i in range(self.num_experts):
            fused = fused + gate[:, i:i+1] * self.experts[i](feat)

        out = self.norm(img + fused)
        return out

class DeformableConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.offset = nn.Conv2d(in_c, 2*9, 3, padding=1)
        self.deform = DeformConv2d(in_c, out_c, 3, padding=1)
        self.norm = LayerNorm2d(out_c)
    def forward(self, x):
        offset = self.offset(x)
        x = self.deform(x, offset)
        return self.norm(x)


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 encoder_channels=256,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.image_encoder = DINOv3(
            backbone=torch.hub.load(
                "/mnt/d/Jialu/buildingage/dinov3",
                'dinov3_vitl16',  # hubconf.py 里定义的函数
                # 'dinov3_vith16plus',  # 1280
                source='local',
                pretrained=False  # 我们用自定义 checkpoint
            ),
            interaction_indexes=[23]
        )
        encoder_channels_list = (encoder_channels, encoder_channels, encoder_channels, encoder_channels)

        for n, value in self.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        self.neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False, ),
            LayerNorm2d(512),
            DeformableConvBlock(512, encoder_channels),
            LayerNorm2d(encoder_channels) )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
            LayerNorm2d(encoder_channels),
            nn.GELU(),
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder = Decoder(encoder_channels_list, decode_channels, dropout, window_size, num_classes)
        self.deep_encoder =DSMEncoder(encoder_channels)
        self.ufz_encoder =LULCEncoder(encoder_channels)

    
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(decode_channels),
            nn.Dropout(0.15),
            nn.Linear(decode_channels, decode_channels),
            nn.GELU(),
            nn.BatchNorm1d(decode_channels),
            nn.Linear(decode_channels, num_classes)
        )
        

         
        self.fuse1 = CosGuidedMoEFusion(encoder_channels)
        self.fuse2 = CosGuidedMoEFusion(encoder_channels)
        self.fuse3 = CosGuidedMoEFusion(encoder_channels)
        self.fuse4 = CosGuidedMoEFusion(encoder_channels)
        self.AdaIN= GeoConditionalAdaIN(img_dim=decode_channels)
        self.attentionpool=AttentionPool(decode_channels)

    def forward(self, x, depth, masks,ufzs,geo_feat=None):
        b, _, h, w = x.size()
        modality_mask = ufzs.flatten(1).all(dim=1, keepdim=True).float()
        ones_b1 = torch.ones(b, 2, device=x.device)
        modality_mask = torch.cat([ones_b1,modality_mask], dim=1)

        depth_feats = self.deep_encoder(depth)
        ufzs_feats = self.ufz_encoder(ufzs)
        deepx = self.image_encoder(x)  # 256*1024  
        deepx = deepx[0].permute(0, 2, 1).view(b, 1024, 32, 32)
        deepx = self.neck(deepx)
        res1 = self.fpn1(deepx)   # 256 128 128 
        res2 = self.fpn2(deepx) # 256 64 64
        res3 = self.fpn3(deepx) # 256 32 32
        res4 = self.fpn4(deepx) # 256 16 16

        res1 = self.fuse1(res1, depth_feats[0], ufzs_feats[0],modality_mask)
        res2 = self.fuse2(res2, depth_feats[1], ufzs_feats[1],modality_mask)
        res3 = self.fuse3(res3, depth_feats[2], ufzs_feats[2],modality_mask)
        res4 = self.fuse4(res4, depth_feats[3], ufzs_feats[3],modality_mask)

        x_piexl,feat_map = self.decoder(res1, res2, res3, res4, h, w)
        # 遍历该图的所有mask

        B, d, W_feat, H_feat = feat_map.shape
        instance_feats=[]
        geos_list=[]
        for b in range(B):
            mask = masks[b, :, :] 
            feature = feat_map[b, :, :, :]
            building_ids = torch.unique(mask)
            building_ids = [id for id in building_ids if id != -1]
            for bid in building_ids:
                mask_bid = (mask == bid).float()  
                mask_bid = mask_bid.unsqueeze(0).unsqueeze(0)
                mask_interp = nn.functional.interpolate(
                    mask_bid, 
                    size=(H_feat,W_feat), 
                    mode='nearest',       
                )
                mask_interp = mask_interp[0]

                pooled_feat = self.attentionpool(feature.unsqueeze(0), mask_interp.unsqueeze(0))
                instance_feats.append(pooled_feat)


                geos_list.append(geo_feat[b,bid].unsqueeze(0))

        attention_f =torch.cat(instance_feats, dim=0)  # [N, C]
        geos=torch.cat(geos_list, dim=0)
        fused_feat = self.AdaIN(attention_f, geos) 
        inst_logits = self.classifier(fused_feat)  # (B×3)×num_classes
        logits_clamped = torch.clamp(inst_logits, min=-10.0, max=10.0)
        return x_piexl, logits_clamped 
    


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os.path
import torch.nn.functional as F


from .cam_mamba import CAM
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
import clip.clip as clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path1 = '/dataset_cc/Pretrain-models/ViT-B-16.pt'  # 不用下载,用下载好的
    model_path2 = '/YCY/Pretrained_models/ViT-B-16.pt'  # 不用下载,用下载好的
    if os.path.exists(model_path1):
        model_path = model_path1
    elif os.path.exists(model_path2):
        model_path = model_path2
    else:
        model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class CLIMB(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(CLIMB, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(self.in_planes_proj+self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)     #冻结了第一层卷积 (conv1) 即 Patch Projection 层的权重
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        self.classifier2 = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.bottleneck_proj_sp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_sp.bias.requires_grad_(False)
        self.bottleneck_proj_sp.apply(weights_init_kaiming)
        self.spatial_branch_name = 'CMIC-CAM'
        self.cam_mamba = CAM(
            dim=768,
            d_state=cfg.MODEL.CAM_D_STATE,
            cluster_num=cfg.MODEL.CAM_CLUSTER_NUM,
            inner_rank=cfg.MODEL.CAM_INNER_RANK,
            mlp_ratio=cfg.MODEL.CAM_MLP_RATIO,
            n_iter=cfg.MODEL.CAM_N_ITER,
            ema_decay=cfg.MODEL.CAM_EMA_DECAY,
        )
        print('Using CMIC-CAM spatial branch with cluster_num={}'.format(cfg.MODEL.CAM_CLUSTER_NUM))
        self.norm2_mamba = nn.LayerNorm(768)
        self.norm3_mamba = nn.LayerNorm(768)
        self.sp_attention = nn.Sequential(
            nn.Linear(768, 192),
            nn.Tanh(),
            nn.Linear(192, 1)
        )

    def forward(self, x, get_image = False, cam_label= None, view_label=None, return_visuals=False):
        if get_image == True:
            if hasattr(self, "cv_embed") and cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif hasattr(self, "cv_embed") and cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif hasattr(self, "cv_embed") and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            _, image_features, image_features_proj, = self.image_encoder(x, cv_embed)
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)

            out_feat = torch.cat([feat, feat_proj], dim=1)
            return out_feat

        if hasattr(self, "cv_embed") and cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif hasattr(self, "cv_embed") and cam_label != None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif hasattr(self, "cv_embed") and view_label != None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        _, image_features, image_features_proj, = self.image_encoder(x, cv_embed)
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        out_feat = torch.cat([feat, feat_proj], dim=1)

        feats_for_mamba = image_features.detach()  # torch.Size([64, 129, 768])
        # BT, hw, D => BT, D, hw => B, T, D, hw => B, D, T, hw => B, D, T, h, w
        # feats_for_mamba = feats_for_mamba.permute(0, 2, 1)  # torch.Size([64, 768, 128])
        feats_for_mamba_sp = feats_for_mamba[:, 1:, :].detach()
        feats_for_mamba_cls = feats_for_mamba[:, 0, :].detach()  # torch.Size([64, 768])
        collect_visuals = return_visuals and not self.training

        if collect_visuals:
            reference_norm = F.normalize(feats_for_mamba_cls, dim=-1).unsqueeze(1)
            patch_norm = F.normalize(feats_for_mamba_sp, dim=-1).transpose(1, 2)
            reorder_sim = torch.bmm(reference_norm, patch_norm).squeeze(1)
            reorder_indices = torch.arange(
                feats_for_mamba_sp.size(1), device=feats_for_mamba_sp.device
            ).unsqueeze(0).expand(feats_for_mamba_sp.size(0), -1)
        else:
            reorder_sim, reorder_indices = None, None

        if collect_visuals:
            mamba_sp_out, cluster_idx = self.cam_mamba(
                feats_for_mamba_sp,
                (self.h_resolution, self.w_resolution),
                return_cluster=True,
            )
        else:
            mamba_sp_out = self.cam_mamba(
                feats_for_mamba_sp,
                (self.h_resolution, self.w_resolution),
            )
            cluster_idx = None
        mamba_sp_out = torch.cat((feats_for_mamba_cls.unsqueeze(1), mamba_sp_out), dim=1)  # torch.Size([64, 129, 768])
        # mamba_sp_out = mamba_sp_out.mean(1)  # torch.Size([64, 768])
        mamba_sp_out2 = self.norm2_mamba(mamba_sp_out)  # bt, 128, 768
        A = self.sp_attention(mamba_sp_out2)  # [B, n, K]  # torch.Size([8, 1024, 1])
        A = torch.transpose(A, 1, 2)  # torch.Size([8, 1, 1024])
        A = F.softmax(A, dim=-1)  # [B, K, n]  # torch.Size([8, 1, 1024])
        mamba_sp_out2 = torch.bmm(A, mamba_sp_out2)  # [B, K, 512]  torch.Size([8, 1, 512])
        mamba_sp_out2 = mamba_sp_out2.squeeze(1)  # torch.Size([64, 768])
        feat_sp = self.bottleneck_proj_sp(mamba_sp_out2)

        if self.training:
            logit = self.classifier(out_feat)
            logitsp = self.classifier2(feat_sp)
            return out_feat, logit, feat_sp, logitsp
        else:
            feat_concat = torch.cat((out_feat, feat_sp), dim=1)
            if return_visuals:
                visual_tensors = {
                    "sim": reorder_sim,
                    "indices": reorder_indices,
                    "attn_weights": A,
                    "branch_name": self.spatial_branch_name,
                    "cluster_idx": cluster_idx,
                    "cluster_num": self.cam_mamba.cluster_num,
                }
                return feat_concat, out_feat, feat_sp, visual_tensors
            return feat_concat, out_feat, feat_sp
            


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue # ignore classifier weights in evaluation
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_classes, camera_num, view_num):
    model = CLIMB(num_classes, camera_num, view_num, cfg)
    return model

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os.path
import torch.nn.functional as F


from .vivim import MambaLayer
#from .spmamba import VSSBlock
from mamba.mamba_ssm.modules.srmamba import SRMamba
from mamba.mamba_ssm.modules.bimamba import BiMamba
from mamba.mamba_ssm.modules.mamba_simple import Mamba
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
        self.sp_mamba_bi = nn.Sequential(
            nn.LayerNorm(768),
            BiMamba(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )
        self.sp_mamba_raw = nn.Sequential(
            nn.LayerNorm(768),
            Mamba(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )
        self.norm2_mamba = nn.LayerNorm(768)
        self.norm3_mamba = nn.LayerNorm(768)
        self.sp_attention = nn.Sequential(
            nn.Linear(768, 192),
            nn.Tanh(),
            nn.Linear(192, 1)
        )
        self.use_attention_reorder = getattr(cfg.MODEL, 'USE_ATTENTION_REORDER', False)
        self.mamba_top_k = getattr(cfg.MODEL, 'MAMBA_TOP_K', 128)
        self.attention_reorder_soft_gate = getattr(cfg.MODEL, 'ATTENTION_REORDER_SOFT_GATE', False)
        self.attention_gate_temperature = getattr(cfg.MODEL, 'ATTENTION_GATE_TEMPERATURE', 0.1)
        self.attention_gate_center = getattr(cfg.MODEL, 'ATTENTION_GATE_CENTER', 0.5)
        if self.use_attention_reorder:
            print('Using attention-guided reordering for Mamba branch')
            if self.attention_reorder_soft_gate:
                print(
                    'Using soft-gated residual attention reorder: temperature={}, center={}'.format(
                        self.attention_gate_temperature, self.attention_gate_center
                    )
                )
        print('Mamba top-k patches: {}'.format(self.mamba_top_k))

    def reorder(self, reference, raw, return_debug=False, top_k=None):
        if top_k is None:
            top_k = raw.size(1)

        reference_norm = F.normalize(reference, dim=-1).unsqueeze(1)  # bt, 1, 768
        raw_norm = F.normalize(raw, dim=-1)  # bt, N, 768
        raw_norm = torch.transpose(raw_norm, 1, 2) # bt, 768, N
        sim = torch.bmm(reference_norm, raw_norm).squeeze(1)  # [bt, 1, 768] [bt, 768, N]= [bt, N]

        _, indices = torch.sort(sim, descending=True)

        # Only keep top-k tokens
        if top_k < indices.size(1):
            indices = indices[:, :top_k]

        selected_patch_embedding = []
        for i in range(indices.size(0)):
            all_patch_embeddings_i = raw[i, :, :].squeeze()
            top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, indices[i])
            top_k_embedding = top_k_embedding.unsqueeze(0)
            selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)

        if return_debug:
            return selected_patch_embedding, sim, indices
        return selected_patch_embedding

    def reorder_by_attention(self, attn_weights, raw, return_debug=False, top_k=None):
        """
        Reorder patch tokens using CLS->patches attention weights from CLIP ViT last layer.
        When soft gate is enabled, the sorted tokens are blended with their original-order
        counterparts to avoid fully discarding spatial order and low-attention identity cues.
        Args:
            attn_weights: (B, 129, 129) attention matrix from last MHA layer
            raw: (B, N, 768) patch token embeddings
            top_k: number of top tokens to keep (default all)
        Returns:
            selected_patch_embedding: (B, top_k, 768) reordered patches
        """
        # Extract CLS token (index 0) attention to all patch tokens (index 1:)
        cls_attn = attn_weights[:, 0, 1:]  # (B, N)
        # Normalize with softmax for stability
        cls_attn = F.softmax(cls_attn, dim=-1)
        _, indices = torch.sort(cls_attn, descending=True)

        # Only keep top-k tokens
        if top_k is not None and top_k < indices.size(1):
            indices = indices[:, :top_k]

        selected_patch_embedding = torch.gather(raw, 1, indices.unsqueeze(-1).expand(-1, -1, raw.size(-1)))
        if self.attention_reorder_soft_gate:
            original_patch_embedding = raw[:, :indices.size(1), :]
            selected_attn = torch.gather(cls_attn, 1, indices)
            attn_min = selected_attn.min(dim=1, keepdim=True)[0]
            attn_max = selected_attn.max(dim=1, keepdim=True)[0]
            selected_attn = (selected_attn - attn_min) / (attn_max - attn_min + 1e-6)
            gate = torch.sigmoid(
                (selected_attn - self.attention_gate_center) / max(self.attention_gate_temperature, 1e-6)
            ).unsqueeze(-1)
            selected_patch_embedding = gate * selected_patch_embedding + (1.0 - gate) * original_patch_embedding

        if return_debug:
            return selected_patch_embedding, cls_attn, indices
        return selected_patch_embedding

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
            _, image_features, image_features_proj = self.image_encoder(x, cv_embed)
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

        collect_visuals = return_visuals and not self.training
        need_attn = self.use_attention_reorder or collect_visuals

        if need_attn:
            encoder_out = self.image_encoder(x, cv_embed, return_attn=True)
            image_features, image_features_proj, attn_weights = encoder_out[1], encoder_out[2], encoder_out[3]
        else:
            _, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            attn_weights = None

        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        out_feat = torch.cat([feat, feat_proj], dim=1)

        feats_for_mamba = image_features.detach()  # torch.Size([64, 129, 768])
        feats_for_mamba_sp = feats_for_mamba[:, 1:, :].detach()
        feats_for_mamba_cls = feats_for_mamba[:, 0, :].detach()  # torch.Size([64, 768])
        #### reorder

        reorder_sim, reorder_indices = None, None
        reorder_sim_raw, reorder_indices_raw = None, None

        if collect_visuals:
            # Always compute original similarity reorder for visualization comparison
            re_order_sim, sim_raw, indices_sim = self.reorder(
                feats_for_mamba_cls, feats_for_mamba_sp, return_debug=True, top_k=self.mamba_top_k
            )
            # Compute attention-based reorder
            if attn_weights is not None:
                re_order_attn, attn_scores, indices_attn = self.reorder_by_attention(
                    attn_weights, feats_for_mamba_sp, return_debug=True, top_k=self.mamba_top_k
                )
            else:
                re_order_attn, attn_scores, indices_attn = re_order_sim, sim_raw, indices_sim

            # Use attention reorder for actual Mamba input
            re_order_mamba_sp = re_order_attn
            reorder_sim = attn_scores
            reorder_indices = indices_attn
            reorder_sim_raw = sim_raw
            reorder_indices_raw = indices_sim
        elif self.use_attention_reorder and attn_weights is not None:
            # Training / inference with attention reorder
            re_order_mamba_sp = self.reorder_by_attention(attn_weights, feats_for_mamba_sp, top_k=self.mamba_top_k)
        else:
            # Original similarity reorder
            re_order_mamba_sp = self.reorder(feats_for_mamba_cls, feats_for_mamba_sp, top_k=self.mamba_top_k)

        B, num_token, D = re_order_mamba_sp.shape
        mamba_sp_out = self.sp_mamba_bi(re_order_mamba_sp)
        mamba_sp_out = torch.cat((feats_for_mamba_cls.unsqueeze(1), mamba_sp_out), dim=1)
        mamba_sp_out2 = self.norm2_mamba(mamba_sp_out)
        A = self.sp_attention(mamba_sp_out2)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        mamba_sp_out2 = torch.bmm(A, mamba_sp_out2)
        mamba_sp_out2 = mamba_sp_out2.squeeze(1)
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
                    "sim_raw": reorder_sim_raw,
                    "indices_raw": reorder_indices_raw,
                    "attn_weights": A,
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

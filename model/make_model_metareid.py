import torch
import torch.nn as nn
from .meta_adapter import MetaAdapter
from .clip import clip
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.context_length = clip_model.context_length
        
        # Get original positional embedding size
        orig_pos_embed = clip_model.positional_embedding
        orig_dim = orig_pos_embed.shape[1]  # Get embedding dimension
        
        # Create new positional embedding for longer sequences
        context_length = 81  # New context length for metadata prompts
        self.positional_embedding = nn.Parameter(
            torch.zeros(context_length, orig_dim, dtype=orig_pos_embed.dtype)
        )
        
        self.register_buffer(
            "attention_mask",
            torch.triu(torch.ones(self.context_length, self.context_length) * float('-inf'), diagonal=1)
        )
        
        # Initialize new positional embeddings
        # Use interpolation for smoother transition
        orig_length = orig_pos_embed.shape[0]
        pos_interpolated = self._interpolate_pos_embedding(
            orig_pos_embed.data,
            orig_length,
            context_length,
            orig_dim
        )
        self.positional_embedding.data.copy_(pos_interpolated)

    def _interpolate_pos_embedding(self, pos_embed, orig_length, new_length, dim):
        """Interpolate positional embedding from original length to new length."""
        pos_embed = pos_embed.unsqueeze(0)  # Add batch dimension
        pos_embed = torch.nn.functional.interpolate(
            pos_embed.permute(0, 2, 1),  # [1, dim, seq_length]
            size=new_length,
            mode='linear',
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 1)  # [1, seq_length, dim]
        return pos_embed.squeeze(0)  # Remove batch dimension

    def forward(self, prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x
    
def _init_additional_pos_embed(pos_embed, old_size, new_size):
    """Initialize additional positional embeddings using interpolation"""
    if new_size <= old_size:
        return

    # Get device of existing embeddings
    dtype = pos_embed.dtype
    device = pos_embed.device
    
    # Create position ids for interpolation
    old_pos_ids = torch.arange(old_size, dtype=torch.float)
    new_pos_ids = torch.arange(new_size, dtype=torch.float)
    
    # Scale position ids to match the range of old embeddings
    new_pos_ids = new_pos_ids * (old_size - 1) / (new_size - 1)
    
    # Interpolate
    new_embeddings = torch.zeros(new_size - old_size, pos_embed.shape[1], dtype=dtype, device=device)
    for idx in range(old_size, new_size):
        pos = new_pos_ids[idx]
        low_idx = int(pos)
        high_idx = min(low_idx + 1, old_size - 1)
        alpha = pos - low_idx
        
        new_embeddings[idx - old_size] = (1 - alpha) * pos_embed.data[low_idx] + alpha * pos_embed.data[high_idx]
    
    # Update remaining positions
    pos_embed.data[old_size:new_size, :] = new_embeddings

class PromptLearner(nn.Module):
    def __init__(self, clip_model, num_classes=56):
        super().__init__()
        n_ctx = 4  # number of context tokens for class
        n_meta_ctx = 16  # number of context tokens for metadata
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device

        # Initialize learnable context vectors
        cls_vectors = torch.empty(num_classes, n_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # Initialize metadata context vectors
        meta_vectors = torch.empty(1, n_meta_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(meta_vectors, std=0.02)
        self.meta_ctx = nn.Parameter(meta_vectors)

        # Register buffers with correct sizes from the beginning
        with torch.no_grad():
            tokenized_prompts = clip.tokenize('photo of a stoat').to(device)
            embedding = clip_model.token_embedding(tokenized_prompts)
            
            # Reserve space for longer sequence
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", torch.zeros(
                1, 81 - 1 - n_ctx - n_meta_ctx, embedding.shape[-1], 
                dtype=embedding.dtype, 
                device=device
            ))
            # Initialize suffix with original values where possible
            if embedding[:, 1:].shape[1] <= self.token_suffix.shape[1]:
                self.token_suffix[:, :embedding[:, 1:].shape[1]] = embedding[:, 1:]

        self.num_classes = num_classes

    def forward(self, label):
        b = label.shape[0]
        
        # Get class-specific context
        cls_ctx = self.cls_ctx[label]
        if cls_ctx.dim() == 2:
            cls_ctx = cls_ctx.unsqueeze(0)
            
        # Expand metadata context for batch
        meta_ctx = self.meta_ctx.expand(b, -1, -1)
        
        # Expand prefix and suffix for batch
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        
        # Combine all parts
        prompts = torch.cat([
            prefix,      # [B, 1, dim]
            cls_ctx,     # [B, n_ctx, dim]
            meta_ctx,    # [B, n_meta_ctx, dim]
            suffix       # [B, rest, dim]
        ], dim=1)
        
        return prompts

class build_transformer_metareid(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_metareid, self).__init__()
        # Basic configurations
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        # Load CLIP model
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = self._load_clip()
        clip_model.to("cuda")

        # Initialize CLIP components
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearner(clip_model, num_classes)
        
        # Ensure parameters require gradients
        for param in self.text_encoder.parameters():
            param.requires_grad_(True)
        for param in self.prompt_learner.parameters():
            param.requires_grad_(True)

        # Initialize MetaAdapter
        self.meta_adapter = MetaAdapter(
            clip_model=clip_model,
            embed_dim=self.in_planes
        )

        # Initialize classifiers
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # Initialize batch normalization layers
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # Initialize camera/view embeddings if needed
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)

    def _load_clip(self):
        url = clip._MODELS[self.model_name]
        model_path = clip._download(url)
        print(f"Loading CLIP model from: {model_path}")
        
        try:
            raw_model = torch.jit.load(model_path, map_location="cpu")
            print("Successfully loaded JIT model")
            state_dict = raw_model.state_dict()
            print("Successfully extracted state dict")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # 打印state_dict的键以进行调试
        print("State dict keys:", state_dict.keys())
        
        # Don't modify the positional embedding here
        # Let TextEncoder handle the positional embedding adaptation
        model = clip.build_model(
            state_dict,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size
        )
        
        # Set context_length after model creation
        model.context_length = 81
        
        return model

    def forward(self, x=None, metadata=None, label=None, cam_label=None, view_label=None, 
            get_image=False, get_text=False):
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts)
            text_features = text_features[torch.arange(text_features.shape[0]), label]
            return text_features

        if get_image:
            if self.model_name == 'RN50':
                _, _, image_features_proj = self.image_encoder(x)
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                _, _, image_features_proj = self.image_encoder(x)
                return image_features_proj[:,0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, 
                image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features,
                image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                index = cam_label * self.view_num + view_label
                if index.max() >= self.cv_embed.size(0) or index.min() < 0:
                    print(f"Invalid cv_embed index: {index}")
                    index = index % self.cv_embed.size(0)  # Adjust index to be within bounds
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        if metadata is not None and self.training:
            img_feature = self.meta_adapter(img_feature, metadata)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

def make_model(cfg, num_class, camera_num=0, view_num=0):
    model = build_transformer_metareid(num_class, camera_num, view_num, cfg)
    return model
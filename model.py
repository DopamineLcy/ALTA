from functools import partial
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed.nn
import torchvision
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block as DecBlock
from einops import rearrange

from util.pos_embed import get_2d_sincos_pos_embed
from cxrbert import CXRBertModel
from loss_functions import ClipLoss, LocalClipLoss
from adapter_block import Block


class ImageProjectionHead(nn.Module):
    """
    Projection head to be used with BERT CLS token, it's similar to `BertPredictionHeadTransform` in HuggingFace library.
    :param config: CXRBertConfig
    :return: (batch_size, output_size)
    """

    def __init__(self, input_dim, proj_dim) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(input_dim, proj_dim)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(proj_dim, eps=1e-12)
        self.dense_to_output = nn.Linear(proj_dim, proj_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class ALTA(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        args=None,
    ):
        super().__init__()

        self.norm_pix_loss = norm_pix_loss
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.missing_token = nn.Parameter(torch.zeros(1, 196, embed_dim).to(torch.float16))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.view_embedding = nn.Parameter(torch.zeros(1, 2, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.bert_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size * 2) ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        self.projection_head = ImageProjectionHead(embed_dim, 128)
        self.local_projection_head = ImageProjectionHead(embed_dim, 128)

        self.args = args
        init_logit_scale = np.log(1 / 0.07)
        init_logit_scale_local1 = np.log(1 / 0.07)
        init_logit_scale_local2 = np.log(1 / 0.07)
        init_logit_scale_local3 = np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_scale_local1 = nn.Parameter(torch.ones([]) * init_logit_scale_local1)
        self.logit_scale_local2 = nn.Parameter(torch.ones([]) * init_logit_scale_local2)
        self.logit_scale_local3 = nn.Parameter(torch.ones([]) * init_logit_scale_local3)

        self.initialize_weights()
        # Bert encoder
        self.bert_encoder = CXRBertModel.from_pretrained("./BiomedVLP-CXR-BERT-specialized")
        self.bert_encoder.local_projection_head.load_state_dict(self.bert_encoder.cls_projection_head.state_dict())
        print("BERT loaded.")

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        ## initialize attn_adapter
        for n, m in self.blocks.named_modules():
            if "attn_adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize mlp_adapter
        for n, m in self.blocks.named_modules():
            if "mlp_adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token", "temporal_embedding", "view_embedding"}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0] * 2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = rearrange(mask, "(b t) n -> b t n", t=4)[:, 0, :]
        ids_restore = rearrange(ids_restore, "(b t) n -> b t n", t=4)[:, 0, :]

        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_encoder(self, img, a_img, l_img, la_img, mask_ratio):
        img_list = [img, a_img, l_img, la_img]
        img_list = [img.unsqueeze(2) for img in img_list]
        x = torch.cat(img_list, dim=2)

        missing_mask = x.sum(dim=[1,3,4]) == 0

        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        # embed patches
        x = self.patch_embed(x)  # shape = [BT, HW, D]

        x[missing_mask.reshape(-1)] = self.missing_token.repeat(x[missing_mask.reshape(-1)].shape[0], 1, 1)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=4)
        x = (
            x
            + self.view_embedding.repeat(1, 2, 1)
            + torch.cat(
                [
                    self.temporal_embedding[:, 0:1, :].repeat(1, 2, 1),
                    self.temporal_embedding[:, 1:, :].repeat(1, 2, 1),
                ],
                dim=1,
            )
        )
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_embed = x[:, 0]
        cls_embed = rearrange(cls_embed, "(b t) c -> b t c", b=B, t=T)  # B, C, T
        
        others_embed = x[:, 1:]
        others_embed = rearrange(others_embed, "(b t) n d -> (b) (t n) d", t=4)  # B, 49*T, C

        latent = torch.cat((cls_embed[:, 0:1, :], rearrange(others_embed, "(b) (t n) d -> (b) t n d", t=4)[:, 0, ::]), dim=1)
        return cls_embed, others_embed, latent, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, batch, mask_ratio=0.75):
        img_big, a_img, l_img, la_img = (
            batch["img"],
            batch["a_img"],
            batch["l_img"],
            batch["la_img"],
        )
        img = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)(img_big)

        inputs, labels, ids, attention_mask = batch["inputs"], batch["labels"], batch["ids"], batch["attention_mask"]

        img_cls_embed, img_others_embed, latent, mask, ids_restore = self.forward_encoder(img, a_img, l_img, la_img, mask_ratio)

        # global loss
        img_cls_embed = img_cls_embed.mean(dim=1)
        img_global_embed = self.projection_head(img_cls_embed)
        img_global_embed = F.normalize(img_global_embed, dim=-1)

        img_feature = self.bert_embed(img_cls_embed)  # B, C

        inputs = torch.cat((ids, inputs), dim=0)
        attention_mask = torch.cat((attention_mask, attention_mask), dim=0)

        txt_embed = self.bert_encoder(img_feature, inputs, attention_mask, output_cls_projected_embedding=True)

        txt_gloal_embed = txt_embed.cls_projected_embedding
        txt_local_embed = txt_embed.local_proected_embedding.transpose(1, 2)
        txt_gloal_embed = F.normalize(txt_gloal_embed, dim=-1)

        loss_func = ClipLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=self.args.rank,
            world_size=self.args.world_size,
            use_horovod=False,
        )

        contrastive_loss = loss_func(img_global_embed, txt_gloal_embed, self.logit_scale.exp())

        # local loss
        img_local_embed = self.local_projection_head(img_others_embed)

        local_loss_func = LocalClipLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=self.args.rank,
            world_size=self.args.world_size,
            use_horovod=False,
        )
        cap_lens = (attention_mask == 0).to(torch.int32).argmax(dim=1).detach()
        cap_lens = cap_lens.masked_fill_(cap_lens == 0, 128) - 1
        local_contrastive_loss, _ = local_loss_func(
            img_local_embed, txt_local_embed, cap_lens, self.logit_scale_local1.exp(), self.logit_scale_local2.exp(), self.logit_scale_local3.exp()
        )

        # img recon loss
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        img_recon_loss = self.forward_loss(img_big, pred, mask)

        # txt recon loss
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(txt_embed.logits.view(-1, self.bert_encoder.config.vocab_size), labels.view(-1))

        return (contrastive_loss, local_contrastive_loss, img_recon_loss, masked_lm_loss), None, None

    def forward_img_feature(self, x, t):
        b = x.shape[0]
        x = x.cuda()

        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.temporal_embedding[:, 0:1, :] + self.view_embedding[:, 0:1, :]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0, :]
        x = self.projection_head(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward_txt_feature(self, x, normalize=False):

        ids, attention_mask = x["input_ids"], x["attention_mask"]
        ids, attention_mask = ids.cuda(), attention_mask.cuda()

        B, C, N = ids.shape

        ids, attention_mask = ids.reshape(-1, ids.shape[2]), attention_mask.reshape(-1, ids.shape[2])

        txt_embed = self.bert_encoder(None, ids, attention_mask, output_cls_projected_embedding=True).cls_projected_embedding
        if normalize:
            txt_embed = F.normalize(txt_embed, dim=-1)
        txt_embed = txt_embed.reshape(B,C,-1)

        return txt_embed

    def forward_txt_feature_single(self, x):

        ids, attention_mask = x["input_ids"], x["attention_mask"]

        txt_embed = self.bert_encoder(None, ids, attention_mask, output_cls_projected_embedding=True).cls_projected_embedding
        txt_embed = F.normalize(txt_embed, dim=-1)

        return txt_embed
    

def alta(**kwargs):
    model = ALTA(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
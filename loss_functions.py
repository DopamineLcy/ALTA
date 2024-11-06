import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch import distributed as dist
from einops import rearrange
from torch.autograd import Variable
from einops import rearrange


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def gather_features_local(
    image_features,
    text_features,
    cap_lens,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
    all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    all_cap_lens = torch.cat(torch.distributed.nn.all_gather(cap_lens), dim=0)

    return all_image_features, all_text_features, all_cap_lens


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class LocalClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels


    def forward(self, img_features, words_emb, cap_lens, temp1, temp2, temp3, agg="sum"):
        all_img_features, all_words_emb, all_cap_lens = gather_features_local(
            img_features, words_emb, cap_lens, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
        )
        batch_size = all_img_features.shape[0]
        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(all_words_emb.shape[0]):
            # Get the i-th text description
            words_num = all_cap_lens[i]
            word = all_words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 512, 128]
            word = word.repeat(batch_size, 1, 1)  # [B, 512, 25]
            context = all_img_features  # [B, 512, 49*T]

            weiContext, attn = attention_fn(word, context, temp1)  # [B, 512, 128], [B, 128, 49*T]

            att_maps.append(attn[i].unsqueeze(0).contiguous())
            word = rearrange(word, "b c w -> (b w) c", b=batch_size, w=words_num)

            weiContext = weiContext.reshape(batch_size * words_num, -1)  # [1200, 768]

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.reshape(batch_size, words_num)

            row_sim = (temp2 * row_sim).exp()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)

        labels = self.get_ground_truth(all_img_features.device, all_img_features.shape[0])

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        return (loss0 + loss1) / 2, att_maps


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    sourceL = context.size(1)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(context, query)
    # --> batch*sourceL x queryL
    attn = attn.reshape(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.reshape(batch_size, sourceL, queryL)
    attn = rearrange(attn, "b s q -> b q s", b=batch_size, s=sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(attn, context)

    return weightedContext, attn


def local_loss(img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):
        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = attention_fn(word, context, temp1)  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(attn[i].unsqueeze(0).contiguous())  # add attention for curr index  [25, 19, 19]
        word = rearrange(word, "b c w -> (b w) c", b=batch_size, w=words_num)

        weiContext = weiContext.reshape(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.reshape(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return (loss0 + loss1) / 2, att_maps


if __name__ == "__main__":
    a = torch.randn(32, 196, 512)
    b = torch.randn(32, 512, 128)
    local_loss(a, b, [i + 10 for i in range(32)])

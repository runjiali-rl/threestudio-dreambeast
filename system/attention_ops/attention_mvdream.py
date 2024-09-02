
import torch
import math
from einops import repeat, rearrange
import torch as th
from mvdream.ldm.modules.diffusionmodules.openaimodel import MultiViewUNetModel, TimestepEmbedSequential, TimestepBlock
from mvdream.ldm.modules.attention import SpatialTransformer3D, BasicTransformerBlock3D, MemoryEfficientCrossAttention, SpatialTransformer, default, exists
from mvdream.ldm.modules.diffusionmodules.util import checkpoint
from mvdream.ldm.interface import LatentDiffusionInterface, DiffusionWrapper
import xformers
from torch.nn.functional import interpolate
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    # import pdb; pdb.set_trace()
    return embedding




def MultiViewUNetForward(self,
                         x,
                         timesteps=None,
                         context=None,
                         y=None,
                         camera=None,
                         num_frames=1,
                         mask=None,
                         token_index=None,
                         cross_attention_scale: float = 1.0,
                         self_attention_scale: float = 1.0, 
                         **kwargs):
    """
    Apply the model to an input batch.
    :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :param num_frames: a integer indicating number of frames for tensor reshaping.
    :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
    """
    assert x.shape[0] % num_frames == 0, "[UNet] input batch size must be dividable by num_frames!"
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    # Add camera embeddings
    if camera is not None:
        assert camera.shape[0] == emb.shape[0]
        emb = emb + self.camera_embed(camera)

    h = x.type(self.dtype)
    for module in self.input_blocks:
        h = module(h,
                   emb,
                   context,
                   num_frames=num_frames,
                   mask=mask,
                   token_index=token_index,
                   cross_attention_scale=cross_attention_scale,
                   self_attention_scale=self_attention_scale,)
        hs.append(h)
    h = self.middle_block(h,
                        emb,
                        context,
                        num_frames=num_frames,
                        mask=mask,
                        token_index=token_index,
                        cross_attention_scale=cross_attention_scale,
                        self_attention_scale=self_attention_scale,)
    for module in self.output_blocks:
        h = th.cat([h, hs.pop()], dim=1)
        h = module(h,
                   emb,
                   context,
                   num_frames=num_frames,
                   mask=mask,
                   token_index=token_index,
                   cross_attention_scale=cross_attention_scale,
                   self_attention_scale=self_attention_scale,)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)



def TimestepEmbedSequentialForward(self,
                                   x,
                                   emb,
                                   context=None,
                                   num_frames=1,
                                   mask=None,
                                   token_index=None,
                                   cross_attention_scale: float = 1.0,
                                   self_attention_scale: float = 1.0,):
    for layer in self:
        if isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer3D):
            x = layer(x,
                      context,
                      num_frames=num_frames,
                      mask=mask,
                      token_index=token_index,
                      cross_attention_scale=cross_attention_scale,
                      self_attention_scale=self_attention_scale,)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context)
        else:
            x = layer(x)
    return x


def SpatialTransformer3DForward(self,
                                x,
                                context=None,
                                num_frames=1,
                                mask=None,
                                token_index=None,
                                cross_attention_scale: float = 1.0,
                                self_attention_scale: float = 1.0,):
    # note: if no context is given, cross-attention defaults to self-attention
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
    if self.use_linear:
        x = self.proj_in(x)
    for i, block in enumerate(self.transformer_blocks):
        x = block(x,
                  context=context[i],
                  num_frames=num_frames,
                  mask=mask,
                  token_index=token_index,
                  cross_attention_scale=cross_attention_scale,
                  self_attention_scale=self_attention_scale,)
    if self.use_linear:
        x = self.proj_out(x)
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
    if not self.use_linear:
        x = self.proj_out(x)
    return x + x_in


def BasicTransformerBlock3DForward(self,
                                   x,
                                   context=None,
                                   num_frames=1,
                                   mask=None,
                                   token_index=None,
                                   cross_attention_scale: float = 1.0,
                                   self_attention_scale: float = 1.0,):
    return checkpoint(self._forward,
                      (x, context, num_frames, mask, token_index, cross_attention_scale, self_attention_scale),
                      self.parameters(),
                      self.checkpoint)


def _BasicTransformerBlock3DForward(self,
                                    x,
                                    context=None,
                                    num_frames=1,
                                    mask=None,
                                    token_index=None,
                                    cross_attention_scale: float = 1.0,
                                    self_attention_scale: float = 1.0,):
    # 3D self-attention
    # maybe we need a mask for the self-attention
    x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
    x = self.attn1(self.norm1(x),
                   mask=mask,
                   self_attention_scale=self_attention_scale,
                   context=context if self.disable_self_attn else None) + x
    # 2D cross-attention for text
    x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
    x = self.attn2(self.norm2(x),
                   context=context,
                   cross_attention_scale=cross_attention_scale,
                   mask=mask,
                   token_index=token_index) + x
    x = self.ff(self.norm3(x)) + x
    return x


def attention(query, key, value,
              attn_mask=None,
              token_index=None,
              p=0.0,
              has_context=True,
              cross_attention_scale: float = 1.0,
              self_attention_scale: float = 1.0,):
    """
    Compute the attention mechanism.
    
    :param query: an [b*f*h x T x d] Tensor of queries.
    :param key: an [b*f*h x T' x d] Tensor of keys.
    :param value: an [b*f*h x T' x d] Tensor of values.
    :param attn_mask: an [b*f*h x  T' x 1] Tensor of masks.
    :param token_index: a index of the token. List of list of indices.
    """
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    attn = query @ key.transpose(-2, -1) # [b*f*h, T, T']
    # attn = F.softmax(attn, dim=-1)

    if attn_mask is not None:
        if has_context:
            for idx, attn_mask_ in enumerate(attn_mask):
                attn_mask_ = attn_mask_/torch.sum(attn_mask_)*torch.sum(torch.ones_like(attn_mask_))
                attn_mask_ = torch.log(attn_mask_ + 1e-9)
                token_indexes_ = token_index[idx]
                for token_index_ in token_indexes_:
                    attn[:, :, token_index_] = attn[:,  :, token_index_] + (attn_mask_.to(attn.device).to(attn.dtype)*cross_attention_scale)
        else:
            for idx, attn_mask_ in enumerate(attn_mask):             
                attn_mask_ = torch.log(attn_mask_ + 1e-9)
                attn= attn + (attn_mask_.to(attn.device).to(attn.dtype)*self_attention_scale)
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    attn = attn @ value
    return attn


def MemoryEfficientCrossAttentionForward(self,
                                         x,
                                         context=None,
                                         mask=None,
                                         token_index=None,
                                         cross_attention_scale: float = 1.0,
                                         self_attention_scale: float = 1.0,):
    """
    Apply the cross-attention mechanism.
    :param x: an [b*f x T x d] Tensor of inputs. (b*f, n*n, d) b: batch, f: frames, n: resolution, d: dimension of the input
    :param context: an [b*f x T' x d] Tensor of context.
    :param mask: a dict with each key corresponding to [f x n' x n'] Tensor of masks.
    :param token_index: a index of the token. List of list of indices.
    :return: an [b*f x T x d] Tensor of outputs.
    """
    has_context = context is not None
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    b, _, _ = q.shape
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, t.shape[1], self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * self.heads, t.shape[1], self.dim_head)
        .contiguous(),
        (q, k, v),
    )

    # attn_map_shape is [b*f*h, T, T'], h is the number of heads

    if mask is not None:
  
        if has_context: # cross attention
 
            input_resolution = torch.sqrt(torch.tensor(x.shape[1])).int()
            mask_list = []
            if isinstance(mask, dict):
                for key_idx, key in enumerate(mask.keys()):
                    single_mask = interpolate(mask[key].unsqueeze(0), size=(input_resolution, input_resolution), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    single_mask = single_mask.view(mask[key].shape[0], -1) # [f, n, n] -> [f, n*n]
                    f = single_mask.shape[0]
                # reshape mask to [f, T] -> [b*f*h, T] -> [b*f*h, T, 1]
                    mask_list.append(single_mask.repeat(int(self.heads*b/f), 1))
        
                out = attention(q, k, v,
                                attn_mask=mask_list,
                                token_index=token_index,
                                has_context=has_context,
                                cross_attention_scale=cross_attention_scale)

        else: # self attention
            f = mask[list(mask.keys())[0]].shape[0]
            input_resolution = torch.sqrt(torch.tensor(x.shape[1])/f).int()
            mask_list = []
            if isinstance(mask, dict):
                for key_idx, key in enumerate(mask.keys()):
                    single_mask = mask[key]
                    single_mask = interpolate(single_mask.unsqueeze(0), size=(input_resolution, input_resolution), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    single_mask = single_mask.view(-1) # [f, n, n] -> [f*n*n]
                    # reshape mask to [f*T] -> [b*h, f*T] -> [b*h, f*T, 1]
                    single_mask = single_mask.unsqueeze(0).repeat(int(self.heads*x.shape[0]), 1)
                    single_mask = single_mask.unsqueeze(2)
               
                    inv_single_mask = (torch.max(single_mask) - single_mask)/(torch.max(single_mask) - torch.min(single_mask))
                    inv_single_mask = inv_single_mask/ torch.sum(single_mask)*torch.sum(torch.ones_like(single_mask))
                    single_mask = single_mask/ torch.sum(single_mask)*torch.sum(torch.ones_like(single_mask))
                    
                    single_mask = single_mask @ single_mask.transpose(1, 2)
                    inv_single_mask = inv_single_mask @ inv_single_mask.transpose(1, 2)
                    
                    mean_single_mask = torch.mean(single_mask)
                    mean_inv_single_mask = torch.mean(inv_single_mask)

           

                    single_mask[single_mask < mean_single_mask] = single_mask[single_mask < mean_single_mask]/10
                    inv_single_mask[inv_single_mask < mean_inv_single_mask] = inv_single_mask[inv_single_mask < mean_inv_single_mask]/10

                    a = torch.max(single_mask)
                    c = torch.min(single_mask)

                    d = torch.max(inv_single_mask)
                    e = torch.min(inv_single_mask)

                    single_mask = torch.sqrt(single_mask + 1e-9)
                    inv_single_mask = torch.sqrt(inv_single_mask + 1e-9)
                    single_mask = (single_mask + inv_single_mask)/2

                    mask_list.append(single_mask)

                out = attention(q, k, v,
                                attn_mask=mask_list,
                                has_context=has_context,
                                self_attention_scale=self_attention_scale)
            

    else:

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)


    out = (
        out.unsqueeze(0)
        .reshape(b, self.heads, out.shape[1], self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, out.shape[1], self.heads * self.dim_head)
    )
    return self.to_out(out)

def DiffusionWrapperForward(self, *args, **kwargs):
    return self.diffusion_model(*args, **kwargs)


def LatentDiffusionInterfaceApplyModel(self,
                                       x_noisy,
                                       t,
                                       cond,
                                       mask=None,
                                       token_index=None,
                                       cross_attention_scale: float = 1.0,
                                       self_attention_scale: float = 1.0, 
                                       **kwargs):
    assert isinstance(cond, dict)
    return self.model(x_noisy, t,
                      mask=mask,
                      token_index=token_index,
                      cross_attention_scale=cross_attention_scale,
                      self_attention_scale=self_attention_scale,
                      **cond,
                      **kwargs)




def set_forward_mvdream(model, current_path=""):
    if model.__class__.__name__ == 'LatentDiffusionInterface':
        # Replace the forward method of the transformer
        if hasattr(model, 'apply_model'):
            model.apply_model = LatentDiffusionInterfaceApplyModel.__get__(model, LatentDiffusionInterface)
            print(f"Replaced forward method in model: LatentDiffusionInterface")
    for name, layer in model.named_children():
        if model.__class__.__name__ == 'DiffusionWrapper':
            # Replace the forward method of the transformer
            if hasattr(model, 'forward'):
                model.forward = DiffusionWrapperForward.__get__(model, DiffusionWrapper)
                print(f"Replaced forward method in model: DiffusionWrapper")
        if model.__class__.__name__ == 'MultiViewUNetModel':
            # Replace the forward method of the transformer
            if hasattr(model, 'forward'):
                model.forward = MultiViewUNetForward.__get__(model, MultiViewUNetModel)
                print(f"Replaced forward method in model: SD3Transformer2DModel")
        if layer.__class__.__name__ == 'TimestepEmbedSequential':
            # Replace the forward method of the transformer
            if hasattr(layer, 'forward'):
                layer.forward = TimestepEmbedSequentialForward.__get__(layer, TimestepEmbedSequential)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
        if layer.__class__.__name__ == 'SpatialTransformer3D':
            # Replace the forward method of the transformer
            if hasattr(layer, 'forward'):
                layer.forward = SpatialTransformer3DForward.__get__(layer, SpatialTransformer3D)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
        if layer.__class__.__name__ == 'BasicTransformerBlock3D':
            # Replace the forward method of the transformer
            if hasattr(layer, '_forward'):
                layer._forward = _BasicTransformerBlock3DForward.__get__(layer, BasicTransformerBlock3D)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
            if hasattr(layer, 'forward'):
                layer.forward = BasicTransformerBlock3DForward.__get__(layer, BasicTransformerBlock3D)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
        if layer.__class__.__name__ == 'MemoryEfficientCrossAttention':
            # Replace the forward method of the transformer
            if hasattr(layer, 'forward'):
                layer.forward = MemoryEfficientCrossAttentionForward.__get__(layer, MemoryEfficientCrossAttention)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")

        new_path = current_path + '.' + name if current_path else name
        set_forward_mvdream(layer, new_path)
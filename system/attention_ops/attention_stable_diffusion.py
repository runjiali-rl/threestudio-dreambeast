

import torch
import torch.nn as nn
from diffusers.utils import logging
from PIL import Image
import torch.nn.functional as F

from transformers import T5TokenizerFast

from typing import Any, Dict, List, Optional, Union
from diffusers.models.attention import Attention
import os
from tqdm import tqdm
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.attention import JointTransformerBlock
import numpy as np
from diffusers import StableDiffusion3Pipeline
import PIL

import inspect
import cv2
import openai

import re
from collections import defaultdict


attn_maps = dict()


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""



class AttnJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attn_map = None
        self.timestep = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        timestep = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        self.timestep = int(timestep[0])
        # cross attention
        attention_scores = torch.matmul(query, encoder_hidden_states_key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(input_ndim, dtype=torch.float32))

        # Apply softmax to get attention weights
        self.attn_map = attention_scores[1].detach().cpu()


        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # import pdb; pdb.set_trace()
        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states



def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output



def SD3TransformerForward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, timestep=timestep
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)



def JointTranformerForward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor, timestep: torch.LongTensor
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        # import pdb; pdb.set_trace()
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
        cross_attention_kwargs = {
            'timestep': timestep
        }
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, **cross_attention_kwargs
        )
        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states



def set_forward_sd3(model, current_path=""):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        # Replace the forward method of the transformer
        if hasattr(model, 'forward'):
            model.forward = SD3TransformerForward.__get__(model, SD3Transformer2DModel)
            print(f"Replaced forward method in model: SD3Transformer2DModel")
    for name, layer in model.named_children():
        # Check if the current layer is the target layer
        if layer.__class__.__name__ == 'JointTransformerBlock':
            # Replace the forward method of the transformer
            if hasattr(layer, 'forward'):
                layer.forward = JointTranformerForward.__get__(layer, JointTransformerBlock)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
            # Replace the __call__ method of the attn processor
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'processor'):
                # layer.attn.processor.__call__ = JointAttnProcessorCall
                # print(f"Replaced __call__ method in layer: {current_path + '.' + name if current_path else name}")
                layer.attn.processor = AttnJointAttnProcessor2_0()
                print(f"Replaced processor in layer: {current_path + '.' + name if current_path else name}")
 

        new_path = current_path + '.' + name if current_path else name
        set_forward_sd3(layer, new_path)



def hook_fn(name,detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
            del module.processor.attn_map

    return forward_hook



def register_cross_attention_hook(transformer):
    # a = list(unet.named_modules())
    for name, module in transformer.named_modules():

        if not name.split('.')[-1].startswith('attn'):
            # only for sd3
            continue

        hook = module.register_forward_hook(hook_fn(name))
    
    return transformer


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids
    tokens = []
    if isinstance(tokenizer, T5TokenizerFast):
        decoder = {v: k.replace("▁", "").lower() for k, v in tokenizer.vocab.items()}

        for text_input_id in text_input_ids[0]:
            token = decoder[text_input_id.item()]
            tokens.append(token)
    else:
        for text_input_id in text_input_ids[0]:
            token = tokenizer.decode(text_input_id.item())
            tokens.append(token)
    return tokens



def resize_and_save(tokenizer, prompt, timestep=None, path=None, max_height=256, max_width=256, save_path='attn_maps'):
    resized_map = None

    if path is None:
        if timestep:
            for path_ in list(attn_maps[timestep].keys()):
                
                value = attn_maps[timestep][path_]
                # value = torch.mean(value,axis=0).squeeze(0)
                vis_seq_len, seq_len = value.shape
                h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
                value = value.view(h, w, seq_len)
                value = value.permute(2,0,1)

                max_height = max(h, max_height)
                max_width = max(w, max_width)
                value = F.interpolate(
                    value.to(dtype=torch.float32).unsqueeze(0),
                    size=(max_height, max_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) # (77,64,64)
                resized_map = resized_map + value if resized_map is not None else value
        else:
            for timestep in tqdm(attn_maps.keys()):
                for path_ in list(attn_maps[timestep].keys()):
                    value = attn_maps[timestep][path_]
                    # value = torch.mean(value,axis=0).squeeze(0)
                    vis_seq_len, seq_len = value.shape
                    h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
                    value = value.view(h, w, seq_len)
                    value = value.permute(2,0,1)

                    max_height = max(h, max_height)
                    max_width = max(w, max_width)
                    value = F.interpolate(
                        value.to(dtype=torch.float32).unsqueeze(0),
                        size=(max_height, max_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                    resized_map = resized_map + value if resized_map is not None else value
                    
    else:
        value = attn_maps[timestep][path]
        value = torch.mean(value,axis=0).squeeze(0)
        seq_len, h, w = value.shape
        max_height = max(h, max_height)
        max_width = max(w, max_width)
        value = F.interpolate(
            value.to(dtype=torch.float32).unsqueeze(0),
            size=(max_height, max_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # (77,64,64)
        resized_map = value

    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # init dirs
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + f'/{timestep}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if path is not None:
        save_path = save_path + f'/{path}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    for i, (token, token_attn_map) in enumerate(zip(tokens, resized_map)):
        if token == bos_token:
            continue
        if token == eos_token:
            break
        token = token.replace('</w>','')
        token = f'{i}_<{token}>.jpg'

        # min-max normalization(for visualization purpose)
        token_attn_map = token_attn_map.numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)

        # save the image
        image = Image.fromarray(normalized_token_attn_map)
        image.save(os.path.join(save_path, token))


def save_by_timesteps_and_path(tokenizer, prompt, max_height, max_width, save_path='attn_maps_by_timesteps_path'):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        for path in attn_maps[timestep].keys():
            resize_and_save(tokenizer, prompt, timestep, path, max_height=max_height, max_width=max_width, save_path=save_path)

def save_by_timesteps(tokenizer, prompt, max_height, max_width, save_path='attn_maps_by_timesteps'):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        resize_and_save(tokenizer, prompt, timestep, None, max_height=max_height, max_width=max_width, save_path=save_path)

def save(tokenizer, prompt, max_height=256, max_width=256, save_path='attn_maps'):
    resize_and_save(tokenizer, prompt, None, None, max_height=max_height, max_width=max_width, save_path=save_path)


def increase_contrast(image, gamma=0.95):
    """
    Increase the contrast of an image using gamma correction.
    
    Parameters:
    image (torch.Tensor): Image tensor to increase contrast.
    gamma (float): Gamma value for correction. Less than 1 increases contrast.
    
    Returns:
    torch.Tensor: Image tensor with increased contrast.
    """
    
    
    # Apply gamma correction
    corrected_img = torch.pow(image, gamma)
    
    
    return corrected_img


def get_attn_maps(prompt: str,
                  tokenizer,
                  normalize: bool = False,
                  max_height: int = 256,
                  max_width: int = 256,
                  save_path:str = None,
                  save_by_timestep: bool = False,
                  save_by_timestep_and_path: bool = False,
                  timestep_start: int = 1001,
                  timestep_end:int = 0,
                  animal_part_list = None):
    # save by timestep and path and save by timestep are mutually exclusive
    assert not (save_by_timestep and save_by_timestep_and_path), \
        "save_by_timestep and save_by_timestep_and_path are mutually exclusive"
    

    selected_path = ["transformer_blocks.11.attn"]


    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if save_by_timestep_and_path:
        for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
            for path in attn_maps[timestep].keys():
                resized_map = None
                time_path_save_path = os.path.join(save_path, f'{timestep}', f'{path}')
                if not os.path.exists(time_path_save_path):
                    os.makedirs(time_path_save_path, exist_ok=True)
 
                value = attn_maps[timestep][path]
                # value = torch.mean(value,axis=0).squeeze(0)
                vis_seq_len, seq_len = value.shape
                h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
                value = value.view(h, w, seq_len)
                value = value.permute(2,0,1)

                max_height = max(h, max_height)
                max_width = max(w, max_width)
                value = F.interpolate(
                    value.to(dtype=torch.float32).unsqueeze(0),
                    size=(max_height, max_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                # get the max length of different tokenizers
                resized_map = resized_map + value if resized_map is not None else value
                max_length = tokenizer.model_max_length

                attn_map_by_token = defaultdict(list)

                # match with tokens
                tokens = prompt2tokens(tokenizer, prompt)
                bos_token = tokenizer.bos_token
                eos_token = tokenizer.eos_token
                
                max_value = torch.max(resized_map[:max_length])
                min_value = torch.min(resized_map[:max_length])
                for i, token in enumerate(tokens):
                    if token == bos_token:
                        continue
                    elif token == eos_token:
                        break

                    elif animal_part_list is not None:
                        for animal_part in animal_part_list:
                            if token in animal_part.split(" "):
                                token_attn_map = resized_map[i]
                                if normalize:
                                    normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                                    normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
                                else:
                                    normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)
                                attn_map_by_token[animal_part].append(normalized_token_attn_map)    
                    else:
                        token_attn_map = resized_map[i] # (token number, h, w)
                        # min-max normalization(for visualization purpose)


                        if normalize:
                            normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                            normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
                        else:
                            normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)

            
                        attn_map_by_token[token].append(normalized_token_attn_map) # need modification
                    
                for token, token_attn_map_list in attn_map_by_token.items():
                    token_attn_map_list = torch.stack(token_attn_map_list)
                    token_attn_map_list = torch.mean(token_attn_map_list, axis=0)
                    if normalize:
                        token_attn_map_list = (token_attn_map_list - torch.min(token_attn_map_list)) / (torch.max(token_attn_map_list) - torch.min(token_attn_map_list))
                        token_attn_map_list = increase_contrast(token_attn_map_list)

                    attn_map_by_token[token] = token_attn_map_list.numpy()
                    if save_path:
                        token = token.replace('</w>','')
                        token = f'<{token}>.jpg'
                        vis_token_attn_map = (token_attn_map_list.numpy() * 255).clip(0, 255).astype(np.uint8)
                        image = Image.fromarray(vis_token_attn_map)
                        image.save(os.path.join(time_path_save_path, token))
                
    if save_by_timestep:
        for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
            resized_map = None
            time_save_path = os.path.join(save_path, f'{timestep}')
            if not os.path.exists(time_save_path):
                os.makedirs(time_save_path, exist_ok=True)
            for path in attn_maps[timestep].keys():
                value = attn_maps[timestep][path]
                # value = torch.mean(value,axis=0).squeeze(0)
                vis_seq_len, seq_len = value.shape
                h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
                value = value.view(h, w, seq_len)
                value = value.permute(2,0,1)

                max_height = max(h, max_height)
                max_width = max(w, max_width)
                value = F.interpolate(
                    value.to(dtype=torch.float32).unsqueeze(0),
                    size=(max_height, max_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                # get the max length of different tokenizers
                resized_map = resized_map + value if resized_map is not None else value
            max_length = tokenizer.model_max_length

            attn_map_by_token = defaultdict(list)

            # match with tokens
            tokens = prompt2tokens(tokenizer, prompt)
            bos_token = tokenizer.bos_token
            eos_token = tokenizer.eos_token
            
            max_value = torch.max(resized_map[:max_length])
            min_value = torch.min(resized_map[:max_length])
            for i, token in enumerate(tokens):
                if token == bos_token:
                    continue
                elif token == eos_token:
                    break

                elif animal_part_list is not None:
                    for animal_part in animal_part_list:
                        if token in animal_part.split(" "):
                            token_attn_map = resized_map[i]
                            if normalize:
                                normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                                normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
                            else:
                                normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)
                            attn_map_by_token[animal_part].append(normalized_token_attn_map)    
                else:
                    token_attn_map = resized_map[i] # (token number, h, w)
                    # min-max normalization(for visualization purpose)


                    if normalize:
                        normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                        normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
                    else:
                        normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)

        
                    attn_map_by_token[token].append(normalized_token_attn_map) # need modification
                
            for token, token_attn_map_list in attn_map_by_token.items():
                token_attn_map_list = torch.stack(token_attn_map_list)
                token_attn_map_list = torch.mean(token_attn_map_list, axis=0)
                if normalize:
                    token_attn_map_list = (token_attn_map_list - torch.min(token_attn_map_list)) / (torch.max(token_attn_map_list) - torch.min(token_attn_map_list))
                    token_attn_map_list = increase_contrast(token_attn_map_list)

                attn_map_by_token[token] = token_attn_map_list.numpy()
                if save_path:
                    token = token.replace('</w>','')
                    token = f'<{token}>.jpg'
                    vis_token_attn_map = (token_attn_map_list.numpy() * 255).clip(0, 255).astype(np.uint8)
                    image = Image.fromarray(vis_token_attn_map)
                    image.save(os.path.join(time_save_path, token))
 

    
    # average over the timesteps that are between timestep_start and timestep_end
    resized_map = None             
    for timestep in tqdm(attn_maps.keys()):
        if timestep > timestep_start or timestep < timestep_end:
            continue
    
        for path_ in list(attn_maps[timestep].keys()):
            if path_ not in selected_path:
                continue
            value = attn_maps[timestep][path_]
            # value = torch.mean(value,axis=0).squeeze(0)
            vis_seq_len, seq_len = value.shape
            h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
            value = value.view(h, w, seq_len)
            value = value.permute(2,0,1)

            max_height = max(h, max_height)
            max_width = max(w, max_width)
            value = F.interpolate(
                value.to(dtype=torch.float32).unsqueeze(0),
                size=(max_height, max_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
            resized_map = resized_map + value if resized_map is not None else value


    # get the max length of different tokenizers

    attn_map_by_token = defaultdict(list)

    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    
    max_value = torch.max(resized_map[:max_length])
    min_value = torch.min(resized_map[:max_length])
    for i, token in enumerate(tokens):
        if token == bos_token:
            continue
        elif token == eos_token:
            break
        elif animal_part_list is not None:
            for animal_part in animal_part_list:
                if token in animal_part.split(" "):
                    token_attn_map = resized_map[i]
                    if normalize:
                        normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                        normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
                    else:
                        normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)
                    attn_map_by_token[animal_part].append(normalized_token_attn_map)    
        else:
            token_attn_map = resized_map[i]
            # min-max normalization(for visualization purpose)
  
            if normalize:
                normalized_token_attn_map = (token_attn_map - torch.min(token_attn_map)) / (torch.max(token_attn_map) - torch.min(token_attn_map))
                normalized_token_attn_map = increase_contrast(normalized_token_attn_map)
            else:
                normalized_token_attn_map = (token_attn_map - min_value) / (max_value - min_value)
            attn_map_by_token[token].append(normalized_token_attn_map)
    for token, token_attn_map_list in attn_map_by_token.items():
        token_attn_map_list = torch.stack(token_attn_map_list)
        token_attn_map_list = torch.mean(token_attn_map_list, axis=0)
        if normalize:
            token_attn_map_list = (token_attn_map_list - torch.min(token_attn_map_list)) / (torch.max(token_attn_map_list) - torch.min(token_attn_map_list))
            token_attn_map_list = increase_contrast(token_attn_map_list)
        attn_map_by_token[token] = token_attn_map_list.numpy()
        if save_path:
            token = token.replace('</w>','')
            token = f'<{token}>.jpg'
            vis_token_attn_map = (token_attn_map_list.numpy() * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(vis_token_attn_map)
            image.save(os.path.join(save_path, token))

            
    

    return attn_map_by_token





def display_sample(image, i, save_dir):
    if isinstance(image, PIL.Image.Image):
        image_pil = image
    else:
        image = image*0.5 + 0.5
        image = image.permute(0, 2, 3, 1)
        image = image.cpu().detach().numpy()
        image_processed = (image * 255).clip(0, 255)
        image_processed = image_processed.astype(np.uint8)

        image_pil = PIL.Image.fromarray(image_processed[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sample_{i}.png')
    image_pil.save(save_path)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    exclude_first: bool = False,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
     
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    if exclude_first:
        timesteps = timesteps[1:]
        scheduler.timesteps = timesteps
        sigmas = sigmas[1:]
        scheduler.sigmas = sigmas

  
    return timesteps, num_inference_steps, sigmas





def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def predict_noise_residual(model,
                           latent_model_input,
                           t,
                           prompt_embeds,
                           timestep_cond,
                           guidance_scale,
                           guidance_rescale,
                           pooled_prompt_embeds=None):
    """
    Predicts the noise residual and performs guidance.

    Args:
        model: The model used to predict the noise residual.
        latent_model_input: The input to the model.
        t: The current timestep.
        prompt_embeds: The prompt embeddings.
        timestep_cond: The timestep condition.
        guidance_scale: The scale for classifier-free guidance.
        guidance_rescale: The rescale value for guidance.

    Returns:
        torch.Tensor: The predicted noise residual after guidance.
    """
    # Predict the noise residual

    assert pooled_prompt_embeds is not None, "pooled_prompt_embeds must be provided for stable_diffusion_3"
    noise_pred = model.transformer(
        hidden_states=latent_model_input,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        # added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    if guidance_rescale > 0.0:
        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

    return noise_pred


def post_process_animal_part_extraction(animal_part):
    # Regular expression to find contents between double quotes
    pattern = r'"(.*?)"'
    # Find all matches
    animal_part_list = re.findall(pattern, animal_part)
    return animal_part_list



def animal_part_extractor(prompt, api_key, max_trial=100):
    with open('custom/threestudio-dreambeast/system/attention_ops/part_extraction_prompt/part_extraction.txt', 'r') as f:
        animal_part_extraction_prompt = f.read()
    animal_part_extraction_prompt = animal_part_extraction_prompt.replace("[COMPOSITE DESCRIPTION]", prompt)
    
    client = openai.OpenAI(
    # This is the default and can be omitted
        api_key=api_key
    )
    trial_idx = 0
    while trial_idx < max_trial:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": animal_part_extraction_prompt,
                }
            ],
            model="gpt-4o",
        )
        animal_parts = chat_completion.choices[0].message.content
        animal_part_list = post_process_animal_part_extraction(animal_parts)
        if len(animal_part_list) > 0 and check_animal_part(animal_part_list, prompt):
            break



    if len(animal_part_list) == 0 or not check_animal_part(animal_part_list, prompt):
        raise ValueError("Failed to extract animal parts from the prompt.")
    return animal_part_list



def check_animal_part(animal_part_list, prompt):
    for animal_part in animal_part_list:
        if not len(animal_part.split(" ")) <= 4:
            return False
        for token in animal_part.split(" "):
            if token not in prompt:
                return False
    
    return True

def extract_attn_maps(
    model: StableDiffusion3Pipeline,
    prompt: str, 
    api_key: str,
    num_inference_steps: int = 50, 
    guidance_scale: float = 7, 
    guidance_rescale: float = 0, 
    device: str = "cuda",  
    interval: int = 10,
    save_dir=None,
    normalize: bool = False,
    image: PIL.Image.Image = None,
    save_by_timestep: bool = False,
    save_by_timestep_and_path: bool = False,
    timestep_start: Optional[int] = 1001,
    timestep_end: Optional[int] = 0,
    free_style_timestep_start: Optional[int] = 501,
    animal_names: List[str] = None,
    ):


    height = model.default_sample_size * model.vae_scale_factor
    width = model.default_sample_size * model.vae_scale_factor


    print("Encoding text prompts")


    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = model.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        # num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=True,
        max_sequence_length=256,
    )
    prompt_embed = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embed = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # Prepare timesteps
    timesteps, num_inference_steps, sigmas = retrieve_timesteps(
        model.scheduler, num_inference_steps, device
    )

    # Prepare latent variables

    num_channels_latents = model.transformer.config.in_channels
    with torch.no_grad():
        noise_latents = model.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embed.dtype,
            device,
            generator=None,
            latents=None,
        )

        if image is not None:
            # if there is an image, we load the image and encode it
            # resize image to height and width
            image = image.resize((width, height))
            image = model.image_processor.preprocess(image).to(device=device).to(noise_latents.dtype)
            image_latents = model.vae.encode(image).latent_dist.sample()

            timesteps, num_inference_steps, sigmas = retrieve_timesteps(
                model.scheduler, num_inference_steps, device, exclude_first=True
            )

    

    # num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
    model._num_timesteps = len(timesteps)

    with torch.no_grad():
        stepped_latents = None
        for i, t in enumerate(tqdm(timesteps)):
            if image is not None and (t>free_style_timestep_start or i==0): # don't do free style
                # if there is already an image, we use the image latents with 
                sigma = sigmas[i]
                latents = sigma * noise_latents + (1 - sigma) * image_latents
            else: # do free style
                # if there is no image or we enter the freestyle zone, we just use the predicted denoised latents
                if stepped_latents is None:
                    latents = noise_latents
                else:
                    latents = stepped_latents
            latent_model_input = torch.cat([latents] * 2)

            time_step = t.expand(latent_model_input.shape[0])
            timestep_cond = None

            noise_pred = predict_noise_residual(
                model,
                latent_model_input,
                time_step,
                prompt_embed,
                timestep_cond,
                guidance_scale,
                guidance_rescale,
                pooled_prompt_embed,
            )


            stepped_latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if i % interval == 0 or i == len(timesteps) - 1:
                vis_image = model.vae.decode(stepped_latents / model.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                if save_dir:
                    display_sample(vis_image, i, save_dir)

    if save_dir:
        attn_map_save_dir = os.path.join(save_dir, "attn_map")
    else:
        attn_map_save_dir = None

    attn_map_by_token = get_attn_maps(
        prompt=prompt,
        tokenizer=model.tokenizer,
        save_path=attn_map_save_dir,
        save_by_timestep=save_by_timestep,
        save_by_timestep_and_path=save_by_timestep_and_path,
        timestep_start=timestep_start,
        timestep_end=timestep_end,
        normalize=normalize,
        animal_part_list=animal_names
    )
    if image is None:
        image = vis_image
    # convert image to numpy array
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().detach().numpy()
    image_processed = (image * 255).clip(0, 255)
    image_processed = image_processed.astype(np.float32)

    # change rgb to bgr
    image = image_processed[..., ::-1]

    vis_image = vis_image.permute(0, 2, 3, 1)
    vis_image = vis_image.cpu().detach().numpy()
    vis_image_processed = (vis_image * 255).clip(0, 255)
    diffused_image = vis_image_processed.astype(np.uint8)



    output = {
        "attn_map_by_token": attn_map_by_token,
        "image": image,
        "diffused_image": diffused_image
    }

    return output





    
    
    
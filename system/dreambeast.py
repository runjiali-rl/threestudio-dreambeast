import os
from dataclasses import dataclass, field

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
from .attention_ops import (extract_attn_maps,
                            set_forward_sd3,
                            register_cross_attention_hook,
                            set_forward_mvdream,
                            animal_part_extractor,
                            prompt2tokens)
from diffusers import DiffusionPipeline
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Dict, Any




@threestudio.register("dreambeast-system")
class DreamBeastSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        prompt_save_path: str = ""
        cache_dir: str = None
        api_key: str = ""

        # global guidance model names
        use_global_attn: bool = False
        global_model_name: str = "stable-diffusion-3-medium-diffusers"
        attention_guidance_start_step: int = 1000
        attention_guidance_timestep_start:int = 850
        attention_guidance_timestep_end:int = 400
        attention_guidance_free_style_timestep_start:int = 500
        record_attention_interval: int = 10
        attntion_nerf_train_interval: int = 2000

        cross_attention_scale: float = 1.0
        self_attention_scale: float = 1.0

        visualize: bool = False
        visualize_save_dir: str = ""

        attention_system: defaultdict = field(default_factory=defaultdict)


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.cache_dir = self.cfg.cache_dir

     
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        set_forward_mvdream(self.guidance.model)
  

        self.attention_guidance_prompt = self.cfg.prompt_processor.prompt
        self.attention_guidance_negative_prompt = self.cfg.prompt_processor.negative_prompt
        attn_file_name = self.attention_guidance_prompt.replace(" ", "_")+"attn.pth"
        self.attn_save_path = os.path.join(self.cache_dir, attn_file_name)
        # initialize the global guidance model
        if not os.path.exists(self.attn_save_path):
            self.global_model = DiffusionPipeline.from_pretrained(self.cfg.global_model_name,
                                                            use_safetensors=True,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=self.cfg.cache_dir)
            set_forward_sd3(self.global_model.transformer)
            register_cross_attention_hook(self.global_model.transformer)
            self.global_model = self.global_model.to("cuda")
            self.global_model.enable_model_cpu_offload()
            self.part_prompts = animal_part_extractor(self.cfg.prompt_processor.prompt,
                                                      api_key=self.cfg.api_key)
        else:
            # this is stupid, you should fix it
            self.global_model = None
            saved_keys = list(torch.load(self.attn_save_path).keys())
            self.part_prompts = [key for key in saved_keys if len(key.split(" ")) <= 4 and \
                                 key not in ["c2w", "width", "height", 
                                            'rays_o',
                                            'rays_d',
                                            'mvp_mtx',
                                            'camera_positions',
                                            'light_positions',
                                            'elevation',
                                            'azimuth',
                                            'camera_distances',
                                            'fovy']]
          
        self.index_by_part = {}

        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.guidance_tokenizer = AutoTokenizer.from_pretrained(self.cfg.prompt_processor.pretrained_model_name_or_path,
                                                                subfolder="tokenizer")
        self.part_token_index_list = []
        for part_prompt in self.part_prompts:
            token_index_list = self.get_token_index(self.guidance_tokenizer, self.cfg.prompt_processor.prompt, part_prompt)
            self.part_token_index_list.append(token_index_list)

        self.attn_map_info = defaultdict(list)
        self.attn_map_batches = defaultdict(list)
        self.attn_map_by_token = defaultdict(list)


        self.attn_geometry = threestudio.find(self.cfg.attention_system.geometry_type)(self.cfg.attention_system.geometry)

        self.attn_material = threestudio.find(self.cfg.attention_system.material_type)(self.cfg.attention_system.material)
        self.attn_background = threestudio.find(self.cfg.attention_system.background_type)(
            self.cfg.attention_system.background
        )
        self.attn_renderer = threestudio.find(self.cfg.attention_system.renderer_type)(
            self.cfg.attention_system.renderer,
            geometry=self.attn_geometry,
            material=self.attn_material,
            background=self.attn_background,
        )

        # iteration setting:
        self.record_attention_start = self.cfg.attention_guidance_start_step
        self.record_attention_end = self.cfg.attention_guidance_start_step + self.cfg.record_attention_interval
        self.attn_nerf_optimize_start = self.cfg.attention_guidance_start_step + self.cfg.record_attention_interval
        self.attn_nerf_optimize_end = self.cfg.attention_guidance_start_step + self.cfg.record_attention_interval + self.cfg.attntion_nerf_train_interval

    

    def get_token_index(self, tokenizer, prompt:str, sub_prompt:str):
        """
        Get the token index of the sub_prompt in the prompt
        args:
        tokenizer: the tokenizer
        prompt: str, the prompt 
        sub_prompt: str, the sub_prompt that we want to find in the prompt

        return:
        token_index_list: List[int], the list of token index
        """
        tokens = prompt2tokens(tokenizer, prompt)
        sub_tokens = prompt2tokens(tokenizer, sub_prompt)
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token
        token_index_list = []
        for idx, token in enumerate(tokens):
            if token == eos_token:
                break
            if token in [bos_token, pad_token]:
                continue
            if token in sub_tokens:
                token_index_list.append(idx)
        return token_index_list
    
                        

    def get_attn_maps(self, images: torch.Tensor):
        """
        Get the attention maps from the global guidance model
        args:

        images: torch.Tensor, shape (B, H, W, 3)

        return:
        attn_map_by_tokens: Dict[str, torch.Tensor], shape (B, num_tokens, H, W)
        """
        
        with torch.no_grad():
         # if the global model is half precision, convert the image to half precision
            view_suffix = ["back view", "side view", "front view", "side view"] * int(images.shape[0] / 4)
            if images is not None:
                images = images.to(self.global_model.dtype)
            else:
                images = [None] * 4
            for idx, image in enumerate(images):
                # convert image to PIL image
                if image is not None:
                    image = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8))
                    if self.cfg.visualize:
                        image.save(os.path.join(self.cfg.visualize_save_dir, f"image_{idx}.png"))

                suffix = view_suffix[idx]
                prompt = self.attention_guidance_prompt + ", " + suffix
                output = extract_attn_maps(model=self.global_model,
                                            prompt=prompt,
                                            animal_names=self.part_prompts,
                                            image=image,
                                            timestep_start=self.cfg.attention_guidance_timestep_start,
                                            timestep_end=self.cfg.attention_guidance_timestep_end,
                                            free_style_timestep_start=self.cfg.attention_guidance_free_style_timestep_start,
                                            save_by_timestep=True,
                                            save_dir=self.cfg.visualize_save_dir,
                                            api_key=self.cfg.api_key,
                                            normalize=True,)
                        
                attn_map_by_token = output['attn_map_by_token']
                
                for key, value in attn_map_by_token.items():
                    value = torch.tensor(value)
                    scale = torch.sum(torch.ones_like(value)) / torch.sum(value)
                    value = value * scale
                    self.attn_map_info[key].append(value)

            del attn_map_by_token



    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        loss = 0.0

        # get the global attention map
        if self.cfg.use_global_attn:
            if not os.path.exists(self.attn_save_path):
                if batch_idx >= self.record_attention_start and batch_idx < self.record_attention_end:
                    out = self(batch)
                    guidance_out = self.guidance(out["comp_rgb"],
                                        self.prompt_utils,
                                        **batch)
                    self.get_attn_maps(images=out["comp_rgb"])
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            for cam2world in value:
                                self.attn_map_info[key].append(cam2world.detach().cpu())
                        
                
                if self.cfg.use_global_attn and batch_idx == self.attn_nerf_optimize_start:
                    for key, value in self.attn_map_info.items():
                        self.attn_map_info[key] = torch.stack(value)
                    
                    attn_file_name = self.attention_guidance_prompt.replace(" ", "_")+"attn.pth"
                    attn_save_path = os.path.join(self.cache_dir, attn_file_name)
                    print("Save the attention map")
                    torch.save(self.attn_map_info, attn_save_path)
                    

            # load the attention map
            if batch_idx == self.attn_nerf_optimize_start:
                print("Load the attention map")
                self.attn_map_info = torch.load(self.attn_save_path)
           
                body_attn_map = None
                for key, value in self.attn_map_info.items():
                    if key in list(batch.keys()):
                        self.attn_map_batches[key] = value
                    elif key in self.part_prompts:
                        #normalize the attention map
                        normalized_value = []
                        for view_idx, attn_map_per_view in enumerate(value):
                            normalized_value_per_view = (attn_map_per_view - torch.min(attn_map_per_view)) / (torch.max(attn_map_per_view) - torch.min(attn_map_per_view))
                            normalized_value.append(normalized_value_per_view)
                        value = torch.stack(normalized_value)
                        self.attn_map_by_token[key] = value
                        if "body" in key:
                            body_attn_map = value
                            body_token = key
                
                # process the attention map specifically for the token that contains body
                if body_attn_map is not None:
                    for key, value in self.attn_map_by_token.items():
                        if key != body_token:
                            quantile = 0.8 # hard coding
                            temp_result = torch.quantile(body_attn_map, quantile, dim=2)

                            # Apply quantile along the first dimension on the result of the previous step
                            quantile_attn_map = torch.quantile(temp_result, quantile, dim=1)
                            selected_idx = value > quantile_attn_map.unsqueeze(1).unsqueeze(2)
                            body_attn_map[selected_idx] = body_attn_map[selected_idx]/1.2 # hard coding
                    
                    self.attn_map_by_token[body_token] = body_attn_map

                        

            # optimize the attention nerf model
            if batch_idx > self.attn_nerf_optimize_start and batch_idx < self.attn_nerf_optimize_end:
                batch_size = batch["c2w"].shape[0]*2
                sample_num = len(self.attn_map_batches[list(self.attn_map_batches.keys())[0]])

                # sample batch_size number of random index
                attn_nerf_batch_idx = np.random.randint(0, sample_num, batch_size)

                attn_batch = {}
                attn_map_gt = []
                for key, value in self.attn_map_batches.items():
                    attn_batch[key] = value[attn_nerf_batch_idx].to(batch["c2w"].device)
                for key, value in self.attn_map_by_token.items():
                    value = value[attn_nerf_batch_idx].to(batch["c2w"].device)
                    # resize the attention map
                    value = F.interpolate(value.unsqueeze(1), (64, 64), mode="bilinear", align_corners=False).squeeze(1)
                    attn_map_gt.append(value)
                attn_map_gt = torch.stack(attn_map_gt, dim=-1)

            
                attn_batch["width"] = batch["width"]
                attn_batch["height"] = batch["height"]

                
                attn_out = self.attn_renderer(**attn_batch)

                attn_map_num = attn_map_gt.shape[-1]
                rendered_attn = attn_out["comp_rgb"][:, :, :, :attn_map_num] 
                # compute softmax over the last dimension
                # rendered_attn = F.softmax(rendered_attn, dim=-1)
                # get the l2 loss
                loss_attn = F.mse_loss(rendered_attn, attn_map_gt)

                loss += loss_attn * self.C(1)

                for view_idx, (_attn_maps, _attn_map_gts) in enumerate(zip(rendered_attn, attn_map_gt)):
                    # save the rendered attention map as Image
                    _attn_maps = _attn_maps.permute(2, 0, 1).detach().cpu().numpy()
                    for part_idx, _attn_map in enumerate(_attn_maps):
                        _attn_map = (_attn_map * 255).astype(np.uint8)
                        _attn_map = cv2.applyColorMap(_attn_map, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(self.cfg.visualize_save_dir, f"attn_{view_idx}_{part_idx}.png"), _attn_map)

                    # save the ground truth attention map as Image
                    _attn_map_gts = _attn_map_gts.permute(2, 0, 1).detach().cpu().numpy()
                    for part_idx, _attn_map_gt in enumerate(_attn_map_gts):
                        _attn_map_gt = (_attn_map_gt * 255).astype(np.uint8)
                        _attn_map_gt = cv2.applyColorMap(_attn_map_gt, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(self.cfg.visualize_save_dir, f"attn_gt_{view_idx}_{part_idx}.png"), _attn_map_gt)
                    
            

            else:
                if batch_idx <= self.attn_nerf_optimize_end:
                    mask = None

                elif batch_idx >= self.attn_nerf_optimize_end:
                    with torch.no_grad():
                        attn_map_num = len(self.part_prompts)
                        rendered_attn = self.attn_renderer(**batch)
                        rendered_attn = rendered_attn["comp_rgb"][:, :, :, :attn_map_num] # 4 x H x W x num_parts
                        rendered_attn = rendered_attn.permute(3, 0, 1, 2) # num_parts x 4 x H x W
                        mask = {}
                        for part_name, attn_map in zip(self.part_prompts, rendered_attn):
                            mask[part_name] = attn_map
                        
    

                out = self(batch)
                guidance_out = self.guidance(out["comp_rgb"],
                            self.prompt_utils,
                            mask = mask,
                            token_index=self.part_token_index_list,
                            cross_attention_scale=self.cfg.cross_attention_scale,
                            self_attention_scale=self.cfg.self_attention_scale,
                            **batch)
                image = out["comp_rgb"]
                if self.cfg.visualize:
                    image = image.detach().cpu().numpy()
                    for idx, img in enumerate(image):
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(os.path.join(self.cfg.visualize_save_dir, f"image_{idx}.png"))
               
        # normal mvdream process
        else:
            out = self(batch)
            guidance_out = self.guidance(out["comp_rgb"],
                                        self.prompt_utils,
                                        **batch)


        
        if batch_idx <= self.attn_nerf_optimize_start or batch_idx >= self.attn_nerf_optimize_end:
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])


            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            if (
                hasattr(self.cfg.loss, "lambda_eikonal")
                and self.C(self.cfg.loss.lambda_eikonal) > 0
            ):
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }   
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )
    

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        self.save_img_sequence(
            f"it{self.true_global_step}-attn-test",
            f"it{self.true_global_step}-attn-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

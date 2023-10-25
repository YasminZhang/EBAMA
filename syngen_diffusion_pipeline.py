import itertools
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import spacy
import torch
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor,

)



import numpy as np
import math
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root

# import image from PIL
from PIL import Image

logger = logging.get_logger(__name__)





class SynGenDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = True,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        # self.doc = ""#self.parser(prompt)

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(
            attention_maps=attention_maps
        )
        return attention_maps_list

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    # Based on StableDiffusionPipeline.__call__ . New code is annotated with NEW.
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            attn_res: Optional[Tuple[int]] = (16, 16),
            syngen_step_size: float = 20.0,
            parsed_prompt: str=None
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            syngen_step_size (`float`, *optional*, default to 20.0):
                Controls the step size of each SynGen update.
            parsed_prompt (`str`, *optional*, default to None):


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # NEW - use parsed_prompt instead of prompt
        if parsed_prompt:
          self.doc = parsed_prompt
        else:
          self.doc = self.parser(prompt)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        negative_prompt_embeds,  prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.stack([negative_prompt_embeds, prompt_embeds], dim=0)
            #print(f"Prompt embeds shape: {prompt_embeds.shape}")

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # NEW - stores the attention calculated in the unet
        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        # NEW
        # text_embeddings = (
        #     prompt_embeds[batch_size * num_images_per_prompt:] if do_classifier_free_guidance else prompt_embeds
        # )
        text_embeddings = [prompt_embeds[1][None,...]]

        if self.model2 is not None:
            negative_prompt_embeds2,  prompt_embeds2 = self._encode_prompt(
            self.prompt2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds_ = torch.stack([negative_prompt_embeds2, prompt_embeds2], dim=0) 

        latents2 = latents.clone().detach().requires_grad_(False)
    
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # NEW
                if self.skip:
                    if i < 25:
                        latents = self._syngen_step(
                            latents,
                            text_embeddings,
                            t,
                            i,
                            syngen_step_size,
                            cross_attention_kwargs,
                            prompt,
                            max_iter_to_alter=25,
                        )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent2_model_input = self.model2.scheduler.scale_model_input(
                    torch.cat([latents2] * 2), t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0] if self.model2 is None else self.model2.unet(
                   torch.stack([latent_model_input[0], latent2_model_input[0], latent_model_input[1], latent2_model_input[1]], dim=0),
                    t,
                    encoder_hidden_states=torch.stack([prompt_embeds[0], prompt_embeds_[0], prompt_embeds[1], prompt_embeds_[1]],dim=0),
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    if self.model2 is None:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                        )
                        noise_pred, noise_pred2 = noise_pred.chunk(2)
                        # noise_pred_uncond, noise_pred_text, noise_pred_uncond2, noise_pred_text2 = noise_pred.chunk(4)
                        # noise_pred = noise_pred_uncond + guidance_scale * (
                        #         noise_pred_text - noise_pred_uncond
                        # )
                        # noise_pred2 = noise_pred_uncond2 + guidance_scale * (
                        #         noise_pred_text2 - noise_pred_uncond2
                        # )
    


                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text[0], guidance_rescale=guidance_rescale)
                    if self.model2 is not None:
                        noise_pred2 = rescale_noise_cfg(noise_pred2, noise_pred_text[1], guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                if self.model2 is None:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents = self.scheduler.step(noise_pred , t, latents, **extra_step_kwargs, return_dict=False)[0]
                    latents2 = self.model2.scheduler.step(noise_pred2, t, latents2, **extra_step_kwargs, return_dict=False)[0] 

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            image2 = self.vae.decode(latents2 / self.vae.config.scaling_factor, return_dict=False)[0]
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None
        
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # convert the torch tensor to a PIL Image
        image2 = self.image_processor.postprocess(image2, output_type=output_type, do_denormalize=do_denormalize)


        # Offload all models
        #self._maybe_free_model_hooks()


        if not return_dict:
            return (image, has_nsfw_concept)
        
        
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        ), StableDiffusionPipelineOutput(
            images=image2, nsfw_content_detected=has_nsfw_concept
        )

    def _syngen_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            cross_attention_kwargs,
            prompt,
            max_iter_to_alter=25,
    ):
        with torch.enable_grad():
            # latents = latents.clone().detach().requires_grad_(True)
            max_iter_to_alter = 1
            
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                # latent = latent.unsqueeze(0)
                # text_embedding = text_embedding.unsqueeze(0)
                latent = latent[None,...]
                

                for k in range(max_iter_to_alter):
                    latent = latent.clone().detach().requires_grad_(True)
                    self.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    )

                    self.unet.zero_grad()
                    # Get attention maps
                    attention_maps = self._aggregate_and_get_attention_maps_per_token()
                    loss = self._compute_loss(attention_maps=attention_maps, prompt=prompt)
            
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size
                        )
                    #print(f"Iteration {i, k} | Loss: {loss:0.4f}")

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents
    

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        attention_for_text = torch.stack(attention_maps, dim=-1)[:,:,1:-1]
        # attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # shift indices by 1 to account for the start token
        indices = [index[-1]-1 if isinstance(index[-1], int) else index[-1][0]-1  for index in self.subtrees_indices] 
      
        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, i]
            smoothing = GaussianSmoothing().to(attention_for_text.device)
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list
    
    @staticmethod
    def _compute_excite_loss(max_attention_per_index: List[torch.Tensor]) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        return loss
    
    def _excitation_loss(self, attention_maps, prompt, attn_map_idx_to_wp):
 
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
        )

        excite_loss = self._compute_excite_loss(max_attention_per_index)

        return excite_loss
    
    def _compute_volumn_attention_per_index(self, attention_maps):
        attention_for_text = torch.stack(attention_maps, dim=-1)[:,:,1:-1]
        # attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # shift indices by 1 to account for the start token
        indices = [index[-1]-1 if isinstance(index[-1], int) else index[-1][0]-1  for index in self.subtrees_indices] 
      
        # Extract the maximum values
        weight_per_token = attention_for_text.sum(dim=[0,1]) / attention_for_text.sum()


        return weight_per_token[indices]

    
    def _sum_loss(self, attention_maps, prompt, attn_map_idx_to_wp):

        volumn_attention_per_index = self._compute_volumn_attention_per_index(
            attention_maps=attention_maps,
        )
        sum_loss = self._compute_excite_loss(volumn_attention_per_index)

        return sum_loss

    def _compute_loss(
            self, attention_maps: List[torch.Tensor], prompt: Union[str, List[str]]
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(self.tokenizer, prompt)
        if self.print_volumn:
            attention_maps_softmax = torch.nn.functional.softmax(torch.stack(attention_maps[1:]), dim=0)
            
            print(f"token {attn_map_idx_to_wp[3]}: {attention_maps_softmax[3-1].sum()/256:0.4f}, {attention_maps_softmax[3-1].max():0.4f} \
                   token {attn_map_idx_to_wp[7]}: {attention_maps_softmax[7-1].sum()/256:0.4f}, {attention_maps_softmax[7-1].max():0.4f}")
        loss = self._attribution_loss(attention_maps, prompt, attn_map_idx_to_wp)

        if self.excite:
            assert hasattr(self, 'lambda_excite'), "please specify lambda_excite"
            loss += self.lambda_excite * self._excitation_loss(attention_maps, prompt, attn_map_idx_to_wp)

        if self.sum_attn:
            assert hasattr(self, 'lambda_sum_attn'), "please specify lambda_sum_attn"
            loss += self.lambda_sum_attn * self._sum_loss(attention_maps, prompt, attn_map_idx_to_wp)

        return loss


    def _attribution_loss(
            self,
            attention_maps: List[torch.Tensor],
            prompt: Union[str, List[str]],
            attn_map_idx_to_wp,
    ) -> torch.Tensor:
        
        self.subtrees_indices = self._extract_attribution_indices(prompt)
        subtrees_indices = self.subtrees_indices
        loss = 0

     

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            positive_loss, negative_loss = self._calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                attn_map_idx_to_wp,
            )
            loss += positive_loss
            loss += negative_loss

        return loss

    def _calculate_losses(
            self,
            attention_maps,
            all_subtree_pairs,
            subtree_indices,
            attn_map_idx_to_wp,
    ):
        positive_loss = []
        negative_loss = []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            positive_loss.append(
                calculate_positive_loss(attention_maps, modifier, noun, dist=self.dist)
            )
            negative_loss.append(
                calculate_negative_loss(
                    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp, dist=self.dist
                )
            )

        positive_loss = sum(positive_loss)
        negative_loss = sum(negative_loss)

        return positive_loss, negative_loss

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text == wp:
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.startswith(wp) and wp != member.text:  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all([wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            paired_indices.append(curr_collected_wp_indices)

        return paired_indices


    def _extract_attribution_indices(self, prompt):
        # extract standard attribution indices
        pairs = extract_attribution_indices(self.doc)

        # extract attribution indices with verbs in between
        pairs_2 = extract_attribution_indices_with_verb_root(self.doc)
        pairs_3 = extract_attribution_indices_with_verbs(self.doc)
        # make sure there are no duplicates
        pairs = unify_lists(pairs, pairs_2, pairs_3)


        #print(f"Final pairs collected: {pairs}")
        paired_indices = self._align_indices(prompt, pairs)
        return paired_indices

def _get_attention_maps_list(
        attention_maps: torch.Tensor
) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list

def is_sublist(sub, main):
    # This function checks if 'sub' is a sublist of 'main'
    return len(sub) < len(main) and all(item in main for item in sub)

def unify_lists(lists_1, lists_2, lists_3):
    unified_list = lists_1 + lists_2 + lists_3
    sorted_list = sorted(unified_list, key=len)
    seen = set()

    result = []

    for i in range(len(sorted_list)):
        if tuple(sorted_list[i]) in seen:  # Skip if already added
            continue

        sublist_to_add = True
        for j in range(i + 1, len(sorted_list)):
            if is_sublist(sorted_list[i], sorted_list[j]):
                sublist_to_add = False
                break

        if sublist_to_add:
            result.append(sorted_list[i])
            seen.add(tuple(sorted_list[i]))

    return result


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype).to(input.device), groups=self.groups)
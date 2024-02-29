import logging
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn
from torchvision.ops import RoIAlign
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    LlamaForCausalLM,
    Blip2PreTrainedModel,
    Blip2VisionModel,
    Blip2Config,
    Blip2QFormerModel,
    GenerationConfig,
)
from transformers.utils import ModelOutput
from transformers.pytorch_utils import apply_chunking_to_forward


logger = logging.getLogger(__name__)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def concat_all_gather(tensor, gather_with_grad=False):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    if gather_with_grad:
        output = torch.cat(torch.distributed.nn.all_gather(tensor), dim=0)  # NOTE: may raise implicit error
    else:
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0)

    return output

@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class AllSeeingModelConfig(Blip2Config):
    def __init__(
        self,
        output_size=(7, 7),
        sampling_ratio=-1,
        aligned=True,
        prompt_length=5,
        lora=False,
        lora_vision=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        self.prompt_length = prompt_length

        self.lora = lora
        self.lora_vision = lora_vision

# Both flatten(Concat) + soft prompt + fix align pos idx
class AllSeeingModel(Blip2PreTrainedModel):
    config_class = AllSeeingModelConfig
    main_input_name = "pixel_values"

    def __init__(self, config: AllSeeingModelConfig):
        super().__init__(config)

        # init vision module
        self.vision_model = Blip2VisionModel(config.vision_config)

        # init region module
        self.grid_size = config.vision_config.image_size // config.vision_config.patch_size
        self.vision_bbox_pos_embed = nn.Parameter(torch.randn(
            1,
            self.grid_size ** 2 + 1,
            config.vision_config.hidden_size,
        ))
        self.vision_bbox_pos_scale = nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
        self.vision_roi_align = RoIAlign(
            output_size=config.output_size,
            spatial_scale=1/config.vision_config.patch_size,
            sampling_ratio=config.sampling_ratio,
            aligned=config.aligned,
        )
        self.vision_projection = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size * config.output_size[0] * config.output_size[1], config.qformer_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.qformer_config.hidden_size, config.qformer_config.hidden_size * config.num_query_tokens),
        )
        # self.vision_roi_align_scale = nn.Parameter(torch.zeros(1, 1, config.qformer_config.hidden_size))

        # init qformer module
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # init language module
        language_model = LlamaForCausalLM(config.text_config)
        self.language_model = language_model
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, language_model.config.hidden_size)

        # init clip module
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_projection = nn.Parameter(torch.empty(self.language_model.config.hidden_size, 512))
        self.image_projection = nn.Parameter(torch.empty(self.language_model.config.hidden_size, 512))
        nn.init.normal_(self.text_projection, std=self.language_model.config.hidden_size ** -0.5)
        nn.init.normal_(self.image_projection, std=self.language_model.config.hidden_size ** -0.5)

        self.align_embedding = nn.Parameter(torch.randn(1, 1, self.language_model.config.hidden_size))
        self.langauge_clip_prompt = nn.Parameter(torch.randn(1, config.prompt_length, self.language_model.config.hidden_size))
        self.langauge_lm_prompt = nn.Parameter(torch.randn(1, config.prompt_length, self.language_model.config.hidden_size))

        self.config.hidden_size = config.text_config.hidden_size
        self.num_queries = config.num_query_tokens
        self.offset = 5

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.lora:
            self.wrap_lora()
        if self.config.lora_vision:
            self.wrap_lora_vision()

        self.gather_with_grad = True
        # self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        self.vision_model.encoder.gradient_checkpointing = True
        self.qformer.encoder.gradient_checkpointing = True
        self.language_model.model.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.vision_model.encoder.gradient_checkpointing = False
        self.qformer.encoder.gradient_checkpointing = False
        self.language_model.model.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def compute_clip_loss(self, image_feats, text_feats):
        image_feats_all = concat_all_gather(image_feats, gather_with_grad=self.gather_with_grad)
        text_feats_all = concat_all_gather(text_feats, gather_with_grad=self.gather_with_grad)

        logit_scale = self.logit_scale.exp()
        sim_i2t = (image_feats @ text_feats_all.t()) * logit_scale
        sim_t2i = (text_feats @ image_feats_all.t()) * logit_scale

        bs = image_feats.size(0)
        device = sim_t2i.device
        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int, device=device)

        # loss = (
        #     F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #     + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        # ) / 2
        loss = (
            F.cross_entropy(sim_i2t, targets)
            + F.cross_entropy(sim_t2i, targets)
        ) / 2

        return loss

    def extract_region_embeds(
        self,
        image_embeds: torch.Tensor,
        boxes: List[torch.Tensor],
    ):
        bsz, _, dim = image_embeds.shape
        image_token_embeds = image_embeds[:, 1:].view(bsz, self.grid_size, self.grid_size, dim).permute(0, 3, 1, 2).contiguous()
        # torch.Size([2, 1408, 16, 16])
        if isinstance(boxes, list):
            boxes = [box.to(torch.float32) for box in boxes]
            image_token_embeds = image_token_embeds.to(torch.float32)
        else:
            boxes = boxes.to(torch.float32)  # torch.Size([2, 10, 4])
            image_token_embeds = image_token_embeds.to(torch.float32)
        boxes = [item for item in boxes]
        region_embeds = self.vision_roi_align(image_token_embeds, boxes)  # bsz, dim, output_size, output_size
        # region_embeds = region_embeds.mean(dim=(-1, -2))
        region_embeds = region_embeds.flatten(1)

        if self.vision_projection[0].weight.dtype in [torch.float16, torch.bfloat16]:
            region_embeds = region_embeds.to(self.vision_projection[0].weight.dtype)
        region_embeds = self.vision_projection(region_embeds)
        region_embeds = region_embeds.view(-1, self.config.num_query_tokens, self.config.qformer_config.hidden_size)

        return region_embeds

    def extract_feature(
            self,
            pixel_values: torch.FloatTensor,
            boxes: List[torch.Tensor],  # (x1,y1,x2,y2)
    ):
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        image_embeds = image_embeds + self.vision_bbox_pos_embed * self.vision_bbox_pos_scale
        region_embeds = self.extract_region_embeds(image_embeds=image_embeds, boxes=boxes)

        # localization info
        # query_tokens = self.query_tokens + region_embeds * self.vision_roi_align_scale
        query_tokens = self.query_tokens.expand(region_embeds.size(0), -1, -1)
        query_tokens = torch.cat([query_tokens, region_embeds], dim=1)

        repeat_time = query_tokens.size(0) // image_embeds.size(0)
        image_embeds = image_embeds.repeat_interleave(repeats=repeat_time, dim=0)
        image_attention_mask = image_attention_mask.repeat_interleave(repeats=repeat_time, dim=0)
        query_outputs = self.qformer(
            query_embeds=query_tokens,  # torch.Size([20, 32, 768])
            encoder_hidden_states=image_embeds,  # torch.Size([20, 257, 1408])
            encoder_attention_mask=image_attention_mask,  # torch.Size([20, 257])
            return_dict=True,
        )

        query_output = query_outputs.last_hidden_state
        del query_outputs
        language_model_inputs = self.language_projection(query_output)

        return language_model_inputs

    def extract_image_feature(
        self,
        pixel_values: torch.FloatTensor,
        boxes: torch.Tensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        language_model_inputs = self.extract_feature(
            pixel_values=pixel_values,  # torch.Size([2, 3, 224, 224])
            boxes=boxes,  # torch.Size([2, 10, 4])
        )  # torch.Size([20, 32, 4096])

        fast_mode = False
        if fast_mode:
            query_feats = language_model_inputs.mean(1)
        else:
            assert torch.all(input_ids[:, self.offset:self.offset + self.num_queries] == 32002)
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            repeat_time = language_model_inputs.size(0) // inputs_embeds.size(0)
            inputs_embeds = inputs_embeds.repeat_interleave(repeats=repeat_time, dim=0)  # torch.Size([20, 52, 4096])
            attention_mask = attention_mask.repeat_interleave(repeats=repeat_time, dim=0)
            # inputs_embeds[:, self.offset:self.offset + self.num_queries, :] = language_model_inputs
            inputs_embeds = torch.cat([
                inputs_embeds[:, :self.offset],
                language_model_inputs,
                inputs_embeds[:, self.offset + self.num_queries:],
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask[:, :self.offset],
                torch.ones((attention_mask.size(0), language_model_inputs.size(1))).to(attention_mask),
                attention_mask[:, self.offset + self.num_queries:],
            ], dim=1)
            
            bsz = inputs_embeds.size(0)
            align_token = self.align_embedding.expand(bsz, -1, -1)
            align_mask = torch.ones(size=(bsz, 1)).to(attention_mask)
            
            prompt_token = self.langauge_clip_prompt.expand(bsz, -1, -1)
            prompt_mask = torch.ones(size=(bsz, prompt_token.size(1))).to(attention_mask)
            
            inputs_embeds = torch.cat([prompt_token, inputs_embeds, align_token], dim=1)
            attention_mask = torch.cat([prompt_mask, attention_mask, align_mask], dim=1)
            
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids[:, -1] = 0
            
            query_feats = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            ).hidden_states[-1]
    
            # seq_len = attention_mask.sum(dim=1)
            # query_feats = query_feats[torch.arange(query_feats.shape[0]), seq_len - 1].squeeze(1)
            query_feats = query_feats[:, -1]

        image_feats = F.linear(query_feats.float(), self.image_projection.t().float())
        image_feats = F.normalize(image_feats, dim=-1)
        return image_feats

    def extract_text_feature(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # input_ids = input_ids.flatten(0, 1)  # torch.Size([80, 8])
        # attention_mask = attention_mask.flatten(0, 1)  # torch.Size([80, 8])

        bsz = input_ids.size(0)
        align_token = self.align_embedding.expand(bsz, -1, -1)
        align_mask = torch.ones(size=(bsz, 1)).to(attention_mask)

        prompt_token = self.langauge_clip_prompt.expand(bsz, -1, -1)
        prompt_mask = torch.ones(size=(bsz, prompt_token.size(1))).to(attention_mask)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prompt_token, inputs_embeds, align_token], dim=1)
        attention_mask = torch.cat([prompt_mask, attention_mask, align_mask], dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids[:, -1] = 0

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # seq_len = attention_mask.sum(dim=1)
        text_feats = outputs.hidden_states[-1]
        del outputs
        # text_feats = text_feats[torch.arange(text_feats.shape[0]), seq_len-1].squeeze(1)
        text_feats = text_feats[:, -1]
        text_feats = F.linear(text_feats.float(), self.text_projection.t().float())
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats

    def forward(self, task='LM', **kwargs):
        if task == 'CLIP':
            return self.clip_forward(**kwargs)

        if task == 'LM':
            return self.lm_forward(**kwargs)

        raise NotImplementedError(f'{task} is not supported for {self.__class__.__name__}')

    def clip_forward(
            self,
            pixel_values: torch.FloatTensor,
            boxes: torch.Tensor,
            query_input_ids: torch.FloatTensor,
            query_attention_mask: Optional[torch.LongTensor],
            label_input_ids: torch.FloatTensor,
            label_attention_mask: Optional[torch.LongTensor],
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,

    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_feats = self.extract_image_feature(
            pixel_values=pixel_values,
            boxes=boxes,
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        text_feats = self.extract_text_feature(
            input_ids=label_input_ids,
            attention_mask=label_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
        )
        loss = self.compute_clip_loss(image_feats=image_feats, text_feats=text_feats)
        if not return_dict:
            return (loss, )

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
        )

    def lm_forward(
            self,
            pixel_values: torch.FloatTensor,
            boxes: List[torch.Tensor],  # (x1,y1,x2,y2)
            input_ids: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,

    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        language_model_inputs = self.extract_feature(pixel_values=pixel_values, boxes=boxes)

        # assert language_model_inputs.shape[1] == self.num_queries
        assert torch.all(input_ids[:, self.offset:self.offset + self.num_queries] == 32002)
        # inputs_embeds[:, self.offset:self.offset + self.num_queries, :] = language_model_inputs
        inputs_embeds = torch.cat([
            inputs_embeds[:, :self.offset],
            language_model_inputs,
            inputs_embeds[:, self.offset + self.num_queries:],
        ], dim=1)
        attention_mask = torch.cat([
            attention_mask[:, :self.offset],
            torch.ones((attention_mask.size(0), language_model_inputs.size(1))).to(attention_mask),
            attention_mask[:, self.offset + self.num_queries:],
        ], dim=1)
        if labels is not None:
            labels = torch.cat([
                labels[:, :self.offset],
                torch.full((attention_mask.size(0), language_model_inputs.size(1)), fill_value=-100).to(attention_mask),
                labels[:, self.offset + self.num_queries:],
            ], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        bsz = inputs_embeds.size(0)
        prompt_token = self.langauge_lm_prompt.expand(bsz, -1, -1)
        prompt_mask = torch.ones(size=(bsz, prompt_token.size(1))).to(attention_mask)

        inputs_embeds = torch.cat([prompt_token, inputs_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        loss = None

        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            logits = logits[:, prompt_token.size(1):, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device).to(torch.long)

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            boxes: List[torch.Tensor],
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            language_model_inputs: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        if language_model_inputs is None:
            language_model_inputs = self.extract_feature(pixel_values=pixel_values, boxes=boxes)

        if input_ids is None:
            batch_size = language_model_inputs.size(1)
            input_ids = torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(batch_size, 1).to(language_model_inputs.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        assert torch.all(input_ids[:, self.offset:self.offset + self.num_queries] == 32002)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # inputs_embeds[:, self.offset:self.offset + self.num_queries, :] = language_model_inputs
        inputs_embeds = torch.cat([
            inputs_embeds[:, :self.offset],
            language_model_inputs,
            inputs_embeds[:, self.offset + self.num_queries:],
        ], dim=1)
        attention_mask = torch.cat([
            attention_mask[:, :self.offset],
            torch.ones((attention_mask.size(0), language_model_inputs.size(1))).to(attention_mask),
            attention_mask[:, self.offset + self.num_queries:],
        ], dim=1)

        bsz = inputs_embeds.size(0)
        prompt_token = self.langauge_lm_prompt.expand(bsz, -1, -1)
        prompt_mask = torch.ones(size=(bsz, prompt_token.size(1))).to(attention_mask)

        inputs_embeds = torch.cat([prompt_token, inputs_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **generate_kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate_cls(self, task='CLIP', **kwargs):
        if task == 'CLIP':
            return self.clip_generate_cls(**kwargs)

        if task == 'LM':
            return self.lm_generate_cls(**kwargs)

        if task == 'ensemble':
            logits_clip = self.clip_generate_cls(**kwargs)
            logits_lm = self.lm_generate_cls(**kwargs)
            return (logits_clip + logits_lm) / 2

        if task == 'all':
            logits_clip = self.clip_generate_cls(**kwargs)
            logits_lm = self.lm_generate_cls(**kwargs)
            return {
                'logits_clip': logits_clip,
                'logits_lm': logits_lm,
                'logits_ensemble': (logits_clip + logits_lm) / 2,
            }

        raise NotImplementedError(f'{task} is not supported for {self.__class__.__name__}')

    @torch.no_grad()
    def clip_generate_cls(
        self,
        pixel_values: torch.FloatTensor,
        boxes: List[torch.Tensor],
        query_input_ids: torch.LongTensor,
        query_attention_mask: Optional[torch.LongTensor],
        label_ids: torch.LongTensor,
        label_mask: Optional[torch.LongTensor],
    ):
        assert torch.all(label_ids[:, 0] == self.config.text_config.bos_token_id)

        image_feats = self.extract_image_feature(
            pixel_values=pixel_values,
            boxes=boxes,
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        # text_feats = self.extract_text_feature(
        #     input_ids=label_ids,
        #     attention_mask=label_mask,
        # )
        if hasattr(self, 'text_feats_cache') and torch.equal(self.text_feats_cache[0], label_ids):
            text_feats = self.text_feats_cache[1]
        else:
            text_feats = []
            chunk_size = 64
            for i in range(0, label_ids.size(0), chunk_size):
                text_feats.append(self.extract_text_feature(
                    input_ids=label_ids[i:i+chunk_size],
                    attention_mask=label_mask[i:i+chunk_size],
                ))
            text_feats = torch.cat(text_feats, dim=0)
            self.text_feats_cache = (label_ids, text_feats)

        logit_scale = self.logit_scale.float().exp()
        logits_per_image = logit_scale * image_feats.float() @ text_feats.t().float()
        return logits_per_image

    def _concat_text_input_output(
        self,
        input_ids,
        input_mask,
        output_ids,
        output_mask,
        ignored_idx=-100,
    ):
        concat_labels = []
        concat_input_ids = []
        concat_attention_mask = []

        for i in range(input_ids.size(0)):
            input_len = input_mask[i].sum()
            concat_input_ids.append(
                torch.cat([
                    input_ids[i][:input_len],
                    output_ids[i],
                    input_ids[i][input_len:],
                ])
            )
            concat_attention_mask.append(
                torch.cat([
                    input_mask[i][:input_len],
                    output_mask[i],
                    input_mask[i][input_len:],
                ])
            )
            concat_labels.append(
                torch.cat([
                    torch.zeros_like(input_ids[i][:input_len]) + ignored_idx,
                    torch.where(output_mask[i].bool(), output_ids[i], ignored_idx),
                    torch.zeros_like(input_ids[i][input_len:]) + ignored_idx,
                ])
            )

        concat_labels = torch.stack(concat_labels)
        concat_input_ids = torch.stack(concat_input_ids)
        concat_attention_mask = torch.stack(concat_attention_mask)
        return concat_input_ids, concat_attention_mask, concat_labels

    @torch.no_grad()
    def lm_generate_cls(
        self,
        pixel_values: torch.FloatTensor = None,
        boxes: List[torch.Tensor] = None,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: Optional[torch.LongTensor] = None,
        label_ids: torch.LongTensor = None,
        label_mask: Optional[torch.LongTensor] = None,
        label_ids_without_role: torch.LongTensor = None,
        label_mask_without_role: Optional[torch.LongTensor] = None,
    ):
        if torch.all(label_ids_without_role[:, 0] == self.config.text_config.bos_token_id):
            label_ids_without_role = label_ids_without_role[:, 1:]
            label_mask_without_role = label_mask_without_role[:, 1:]

        bsz = query_input_ids.size(0)
        num_classes = label_ids_without_role.size(0)

        repeated_boxes = []
        for box in boxes:
            repeated_boxes.extend([box] * num_classes)
        repeated_boxes = torch.stack(repeated_boxes, dim=0)

        num_boxes_per_img = repeated_boxes.size(1)
        repeated_boxes = repeated_boxes.flatten(0, 1).unsqueeze(1)
        repeated_pixel_values = pixel_values.repeat_interleave(repeats=num_classes * num_boxes_per_img, dim=0)
        repeated_query_input_ids = query_input_ids.repeat_interleave(repeats=num_classes, dim=0)
        repeated_query_attention_mask = query_attention_mask.repeat_interleave(repeats=num_classes, dim=0)

        repeated_label_ids = label_ids_without_role.repeat(bsz, 1)
        repeated_label_mask = label_mask_without_role.repeat(bsz, 1)

        input_ids, attention_mask, labels = self._concat_text_input_output(
            input_ids=repeated_query_input_ids,
            input_mask=repeated_query_attention_mask,
            output_ids=repeated_label_ids,
            output_mask=repeated_label_mask,
        )

        # extract logits
        logits = apply_chunking_to_forward(
            self.lm_generate_forward_chunk,
            bsz,
            0,
            repeated_pixel_values,
            repeated_boxes,
            torch.ones_like(repeated_boxes).bool(),
            input_ids,
            attention_mask,
        )
        logits = logits[:, self.config.prompt_length:]

        labels = torch.cat([
            labels[:, :self.offset],
            torch.full((attention_mask.size(0), self.num_queries + logits.size(1) - labels.size(1)), fill_value=-100).to(labels),
            labels[:, self.offset + self.num_queries:],
        ], dim=1)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device).to(torch.long)

        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        # loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        loss = apply_chunking_to_forward(
            loss_fct,
            bsz,
            0,
            shift_logits.view(-1, self.config.text_config.vocab_size),
            shift_labels.view(-1)
        )

        seq_len = logits.size(1)
        cls_logits = loss.view(bsz * num_classes, seq_len-1).mean(dim=1).view(bsz, num_classes)
        cls_logits = (-cls_logits).softmax(-1)
        return cls_logits

    def lm_generate_forward_chunk(self, pixel_values, boxes, input_ids, attention_mask):
        return self(
            pixel_values=pixel_values,
            boxes=boxes,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    @torch.no_grad()
    def simplified_lm_generate_cls(
        self,
        pixel_values: torch.FloatTensor = None,
        boxes: List[torch.Tensor] = None,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: Optional[torch.LongTensor] = None,
        label_ids: torch.LongTensor = None,
        label_mask: Optional[torch.LongTensor] = None,
    ):
        if torch.all(label_ids[:, 0] == self.config.text_config.bos_token_id):
            label_ids = label_ids[:, 1:]
            label_mask = label_mask[:, 1:]
        # label_ids: num_classes, num_tokens
        # label_mask: num_classes, num_tokens
        label_mask = label_mask.bool()

        # extract logits
        logits = self(
            pixel_values=pixel_values,
            boxes=boxes,
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        ).logits  # B, N, Vocab_Size
        logits = logits[:, -1]  # B, Vocab_Size
        logits = logits[:, label_ids].squeeze(1)

        # aggregate prob of each token: (B, num_classes, num_tokens)
        logits = torch.where(label_mask, logits, -torch.inf)
        logits = F.log_softmax(logits, dim=1)
        logits = torch.where(label_mask, logits, 0)
        # cls_logits: (B, num_classes)
        cls_logits = logits.sum(dim=-1) / label_mask.sum(dim=-1)

        return cls_logits

    def wrap_lora(
        self,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.language_model = get_peft_model(self.language_model, peft_config)
        self.config.lora = True
        self.language_model.print_trainable_parameters()

    def wrap_lora_vision(
        self,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("qkv", "projection"),
    ):
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.vision_model = get_peft_model(self.vision_model, peft_config)
        self.config.lora_vision = True
        self.vision_model.print_trainable_parameters()

class AllSeeingModelForCaption(AllSeeingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.logit_scale
        del self.text_projection
        del self.image_projection
        del self.align_embedding
        del self.langauge_clip_prompt

"""CoAD: Automatic Diagnosis through Symptom and Disease Co-Generation
"""
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from utils import getlogger
from models.net import MLP
from typing import List, Optional, Tuple
from torch import nn
import torch
import os

logger = getlogger(name=__name__)


class MyGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.h = nn.ModuleList(
            [
                GPT2Block(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the"
                " same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            ## My Code here
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            else:
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]
            ## My Code end

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=self.dtype
            )  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype
            ).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if (
            self.config.add_cross_attention
            and encoder_hidden_states is not None
        ):
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size,
                encoder_sequence_length,
            )
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device
                )
            encoder_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            ()
            if output_attentions and self.config.add_cross_attention
            else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device)
                        for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient"
                        " checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )


class COAD(nn.Module):
    def __init__(
        self,
        n_syms: int,
        n_dis: int,
        dis_hidden_sizes: List[int] = [],
        dis_dropout_p: float = 0,
        sym_hidden_sizes: List[int] = [],
        sym_dropout_p: float = 0,
        emb_dropout_p: float = 0.1,
    ):
        super().__init__()
        config = GPT2Config.from_json_file(
            os.path.join(os.path.dirname(__file__), "config.json")
        )
        self.model = MyGPT2Model(config=config)
        self.sym_emb = nn.Embedding(n_syms, config.n_embd)
        # 0 unk 1 pos 2 neg
        self.sym_type_emb = nn.Embedding(3, config.n_embd)
        self.sym_head = MLP(
            input_dim=config.n_embd,
            output_dim=n_syms,
            hidden_sizes=sym_hidden_sizes,
            dropout_p=sym_dropout_p,
        )
        self.dis_head = MLP(
            input_dim=config.n_embd,
            output_dim=n_dis,
            hidden_sizes=dis_hidden_sizes,
            dropout_p=dis_dropout_p,
        )
        self.emb_dropout = nn.Dropout(p=emb_dropout_p)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input_ids: torch.tensor,
        sym_type_ids: torch.tensor,
        masks: torch.tensor = None,
        sym_ids: torch.tensor = None,
        sym_weights: torch.tensor = None,
        step_dis_ids: torch.tensor = None,
        step_dis_weights: torch.tensor = None,
        **kwargs
    ):
        """
        Args:
            input_ids (torch.tensor): (B, T)
            sym_type_ids (torch.tensor): (B, T)
            masks (torch.tensor): (B, T, T) or (B, T)
            sym_ids (torch.tensor): (B, T)
            sym_weights (torch.tensor): (B, T)
            step_dis_ids (torch.tensor): (B, T)
            step_dis_weights (torch.tensor): (B, T)
        """
        sym_embeds = self.sym_emb(input_ids)
        sym_type_embeds = self.sym_type_emb(sym_type_ids)
        inputs_embeds = self.emb_dropout(sym_embeds + sym_type_embeds)
        if masks is None:
            masks = torch.ones_like(input_ids)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=masks,
            return_dict=False,
        )
        # shape: (B, T, H)
        hidden_states = outputs[0]
        sym_loss = None
        dis_loss = None
        sym_logits = self.sym_head(hidden_states)
        dis_logits = self.dis_head(hidden_states)
        if sym_ids is not None:
            shift_sym_logits = sym_logits[..., :-1, :].contiguous()
            shift_sym_ids = sym_ids[..., 1:].contiguous()
            shift_sym_weights = sym_weights[..., 1:].contiguous()
            sym_losses = self.loss_fn(
                shift_sym_logits.reshape(-1, shift_sym_logits.shape[2]),
                shift_sym_ids.reshape(-1),
            )
            sym_loss = (sym_losses * shift_sym_weights.reshape(-1)).mean()
        if step_dis_ids is not None:
            dis_losses = self.loss_fn(
                dis_logits.reshape(-1, dis_logits.shape[2]),
                step_dis_ids.reshape(-1),
            )
            dis_loss = (dis_losses * step_dis_weights.reshape(-1)).mean()

        return (sym_loss, dis_loss, sym_logits, dis_logits) + outputs

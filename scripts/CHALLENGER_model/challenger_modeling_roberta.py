import math
from typing import List, Optional, Tuple, Union
import pdb
import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaModel,RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from transformers.activations import ACT2FN, gelu
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    logging,
    replace_return_docstrings,
)
from challenger_roberta_outputs import CHALLENGER_MaskedLMOutput

class CHALLENGER_Roberta_LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.decoder = nn.Linear(config.hidden_size, 3) # MY_UPDATE, OUTPUT: nocall,dup,del
        self.bias = nn.Parameter(torch.zeros(3)) # MY_UPDATE,
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x
    
    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class CHALLENGER_Roberta_CLSHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convLayer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5,padding="same",padding_mode="zeros")
        self.layer_norm = nn.LayerNorm(1002, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(1002, 3)
        self.decoder.bias = nn.Parameter(torch.zeros(3))

    def forward(self, features, attention_mask, **kwargs):
        x = self.convLayer(features).squeeze()
        x = gelu(x)
        x = self.layer_norm(x)
        x = x * attention_mask
        x = self.decoder(x)
        return x

class CHALLENGER_Roberta_CLSHead_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )
        self.mlp_head2 = nn.Linear(config.hidden_size, 3)

    def forward(self, features, attention_mask, **kwargs):
        out_1 = self.mlp_head(features)
        out_2 = self.mlp_head2(out_1)
        return out_2

    
class CHALLENGER_roberta_embeddings(RobertaEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.gene_embeddings = nn.Embedding(config.gene_count, config.hidden_size) # MY_UPDATE
        self.input_embed_layer_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5,padding="same",padding_mode="zeros")
        self.input_embed_layer_2 = nn.Conv1d(in_channels=32, out_channels=config.hidden_size, kernel_size=5,padding="same",padding_mode="zeros")

        self.cov_mean = config.coverage_stats[0]
        self.cov_std = config.coverage_stats[1]

        if(config.baseline_coverages_path!=""):
            self.baseline_coverages_lookup_tensor = torch.load(config.baseline_coverages_path)
            self.baseline_coverages_lookup_tensor = self.baseline_coverages_lookup_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.baseline_coverages_lookup_tensor = None

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, gene_ids=None, label_ids=None, inputs_embeds=None, past_key_values_length=0
    ):


        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :(seq_length)] 
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], (seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # BASELINE_COVERAGES 
        #### MY_UPDATE
        if(self.baseline_coverages_lookup_tensor!=None):
            self.baseline_coverages_lookup_tensor = self.baseline_coverages_lookup_tensor.to(input_ids.device)
            baseline_coverages = self.baseline_coverages_lookup_tensor[gene_ids]
        ####

        if inputs_embeds is None:
            #### MY_UPDATE
            ## alt_1
            input_ids_normalized = input_ids.clone()
            input_ids_normalized[:,1:-1] = input_ids_normalized[:,1:-1]-baseline_coverages
            input_ids_normalized = input_ids_normalized / self.cov_std
            ##
            ### alt_2
            """"
            input_ids_normalized = input_ids.clone().to(torch.float)
            input_ids_normalized[:,1:-1] = input_ids_normalized[:,1:-1]/baseline_coverages
            input_ids_normalized[:,1:-1] = torch.clamp(torch.log2(input_ids_normalized[:, 1:-1]), min=-2.0, max=2.0)
            input_ids_normalized[:,1:-1] = torch.nan_to_num(input_ids_normalized[:,1:-1], nan=1.0)
            """
            ###
            inputs_embeds = input_ids_normalized
            inputs_embeds = inputs_embeds[:, None, :]
            #inputs_embeds = self.input_embed_layer(inputs_embeds)
            inputs_embeds = self.input_embed_layer_1(inputs_embeds)
            inputs_embeds = self.input_embed_layer_2(inputs_embeds)
            inputs_embeds = inputs_embeds.transpose(1, 2)
            inputs_embeds[:,0,:] = torch.zeros((inputs_embeds.shape[0], inputs_embeds.shape[2]))
            inputs_embeds[:,-1,:] = torch.zeros((inputs_embeds.shape[0], inputs_embeds.shape[2]))
            ####
            #inputs_embeds = self.word_embeddings(input_ids)
        
        
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #embeddings = inputs_embeds + token_type_embeddings
        embeddings = inputs_embeds
        
        ##### MY_UPDATE - GENE_EMBEDDING
        gene_embeds = self.gene_embeddings(gene_ids)
        gene_embeds = gene_embeds.unsqueeze(1)
        gene_embeds = gene_embeds.expand(gene_embeds.shape[0],input_shape[1],gene_embeds.shape[2])
        
        
        embeddings += gene_embeds
        #####

        ##### MY_UPDATE
        cnv_label_embeds = self.word_embeddings(label_ids)
        embeddings += cnv_label_embeds
        #####
        
                
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CHALLENGER_roberta_model(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=True)
        self.embeddings = CHALLENGER_roberta_embeddings(config)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        gene_ids: Optional[torch.LongTensor] = None, # MY_UPDATE
        label_ids : Optional[torch.LongTensor] = None, # MY_UPDATE
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            gene_ids=gene_ids, #MY_UPDATE
            label_ids=label_ids, #MY_UPDATE
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class CHALLENGER_roberta_ForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = CHALLENGER_roberta_model(config, add_pooling_layer=False)
        self.lm_head = CHALLENGER_Roberta_LMHead(config) # Segmentation Head
        self.cls_head = CHALLENGER_Roberta_CLSHead_v2(config)  #Classification Head
        
        """
        for param in self.roberta.parameters():
            param.requires_grad = False

        for param in self.lm_head.parameters():
            param.requires_grad = False
        """

        # Initialize weights and apply final processing
        #self.post_init() 
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        gene_ids: Optional[torch.LongTensor] = None, #MY_UPDATE
        label_ids : Optional[torch.LongTensor] = None, # MY_UPDATE
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CHALLENGER_MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            gene_ids=gene_ids, # MY_UPDATE
            label_ids = label_ids, # MY_UPDATE
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        seg_scores = self.lm_head(sequence_output)
        cls_scores = self.cls_head(sequence_output[:,0,:],attention_mask)

        masked_lm_loss = None
        if labels is not None: # v8 uses compute_loss function, that why it is returning none
            # move labels to correct device to enable model parallelism
            labels = labels.to(seg_scores.device)
            loss_fct = CrossEntropyLoss()
            labels_tmp = labels.view(-1)
            positive_flag = (labels_tmp>0)
            labels_tmp[positive_flag] = self.config.nocall_token_id- labels_tmp[positive_flag] 
            masked_lm_loss = loss_fct(seg_scores.view(-1, 3), labels_tmp) # MY_UPDATE 3 : NOCALL, DUP, DEL

        if not return_dict:
            output = (seg_scores,cls_scores) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return CHALLENGER_MaskedLMOutput(
            loss=masked_lm_loss,
            seg_logits=seg_scores,
            cls_logits=cls_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

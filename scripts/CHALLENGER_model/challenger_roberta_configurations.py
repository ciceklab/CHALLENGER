
from transformers import RobertaConfig



class CHALLENGER_roberta_config(RobertaConfig):
   def __init__(
        self,
        vocab_size=50265,
        gene_count = 100, #MY_UPDATE
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        baseline_coverages_path = None, #MY_UPDATE
        nocall_token_id = -1, #MY_UPDATE
        coverage_stats = (0,0), #MY_UPDATE
        **kwargs,
    ):
      super().__init__(vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=intermediate_size,
                        hidden_act=hidden_act,
                        hidden_dropout_prob=hidden_dropout_prob,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        max_position_embeddings=max_position_embeddings,
                        type_vocab_size=type_vocab_size,
                        initializer_range=initializer_range,
                        layer_norm_eps=layer_norm_eps,
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id,
                        position_embedding_type=position_embedding_type,
                        use_cache=use_cache,
                        classifier_dropout=classifier_dropout
                        )
        
      self.gene_count = gene_count
      self.baseline_coverages_path = baseline_coverages_path
      self.nocall_token_id = nocall_token_id
      self.coverage_stats = coverage_stats

# Model architecture definitions.
import torch
import torch.nn as nn

from dataset import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
from typing import List

################################################################################################################
#
#                                         Seq2seq and Output Layer 
#
################################################################################################################
class Seq2Seq(nn.Module):
    def __init__(self,
        enc_num_layers: int,
        src_vocab_size: int,
        enc_embedding_dim: int,
        enc_hidden_dim: int,
        enc_num_heads: int,
        enc_ffn_dim: int,

        dec_num_layers: int,
        trg_vocab_size: int,
        dec_embedding_dim: int,
        dec_hidden_dim: int,
        dec_num_heads: int,
        dec_ffn_dim: int,

        *,
        enc_dropout_prob: float = 0.1,
        enc_max_seq_len: int = 512,
        dec_dropout_prob: float = 0.1,
        dec_max_seq_len: int = 512
    ) -> None:
        super().__init__()

        self.dec_max_seq_len = dec_max_seq_len  # for translate

        # For a vanilla Transformer architecture, all model dimensions must match
        #      (embedding_dim = hidden_dim) for both encoder and decoder
        # 
        # Otherwise, we need additional linear layers to project between different dimensions,
        # or that Add and Norm layers will not work because the input dimensions do not match
        if not (enc_embedding_dim == enc_hidden_dim == dec_embedding_dim == dec_hidden_dim):
            raise ValueError(
                "All model dimensions must match for a vanilla Transformer: "
                f"enc_embedding_dim={enc_embedding_dim}, "
                f"enc_hidden_dim={enc_hidden_dim}, "
                f"dec_embedding_dim={dec_embedding_dim}, "
                f"dec_hidden_dim={dec_hidden_dim}"
            )

        # Encoder
        self.encoder = Encoder(
            num_layers = enc_num_layers,
            src_vocab_size = src_vocab_size,
            embedding_dim = enc_embedding_dim,
            hidden_dim = enc_hidden_dim,
            num_heads = enc_num_heads,
            ffn_dim = enc_ffn_dim,
            dropout_prob = enc_dropout_prob,
            max_seq_len = enc_max_seq_len
        )

        # Decoder
        self.decoder = Decoder(
            num_layers = dec_num_layers,
            trg_vocab_size = trg_vocab_size,
            embedding_dim = dec_embedding_dim,
            hidden_dim = dec_hidden_dim,
            num_heads = dec_num_heads,
            ffn_dim = dec_ffn_dim,
            dropout_prob = dec_dropout_prob,
            max_seq_len = dec_max_seq_len
        )

        # Output Layer (Linear layer to project decoder output to target vocabulary size)
        self.output_layer = OutputLayer(
            hidden_dim = dec_hidden_dim,
            trg_vocab_size = trg_vocab_size
        )

    def forward(self,
        src: torch.Tensor, 
        trg: torch.Tensor,
    ) -> torch.Tensor:

        # [1] Create encoder self attention mask (PADDING MASK)
                                    # enc_self_attention_mask: [batch_size, 1, 1, src_seq_len]  - Encoder mask (PADDING MASK)
        enc_self_attention_mask = self.create_enc_self_attention_mask(src = src)

        # [2] Encoder forward pass
                                    # encoder_out: [batch_size, src_seq_len, hidden_dim]
        encoder_out = self.encoder(
            src = src, 
            mask = enc_self_attention_mask
        )

        # [3] Create decoder input mask and cross attention mask (Combination of PADDING MASK and CAUSAL MASK)
                                    # dec_self_attention_mask: [batch_size, 1, trg_q_len, trg_k_len]  - Decoder mask (Combination of PADDING MASK and CAUSAL MASK)
                                    # dec_cross_attention_mask: [batch_size, 1, 1, src_seq_len]  - Cross-Attention mask (PADDING MASK)
        dec_self_attention_mask = self.create_dec_self_attention_mask(trg = trg)
        dec_cross_attention_mask = self.create_dec_cross_attention_mask(src = src)

        # [4] Decoder forward pass (with encoder output and decoder mask)
                                    # decoder_out: [batch_size, trg_seq_len, hidden_dim]

        decoder_out = self.decoder(
            trg = trg,
            encoder_out = encoder_out,
            dec_self_attention_mask = dec_self_attention_mask,
            dec_cross_attention_mask = dec_cross_attention_mask
        )

        # [5] Output Layer forward pass (project decoder output to target vocabulary size)
                                    # out: [batch_size, trg_seq_len, trg_vocab_size]
        out = self.output_layer(decoder_out)

        return out

    # Translate at inference time - given a source sentence, generate the target sentence
    def translate(self,
        src: torch.Tensor,
        max_len: int = 100
    ) -> List[int]:
        
        self.eval()

        with torch.no_grad():

            # [1] Create encoder self attention mask (PADDING MASK)
                                        # enc_self_attention_mask: [batch_size, 1, 1, src_seq_len]  - Encoder mask (PADDING MASK)
            enc_self_attention_mask = self.create_enc_self_attention_mask(src = src)

            # [2] Encoder forward pass
                                        # encoder_out: [batch_size, src_seq_len, hidden_dim]
            encoder_out = self.encoder(
                src = src, 
                mask = enc_self_attention_mask
            )

            # [3] Create decoder cross attention mask (PADDING MASK) - only need to do this once
            dec_cross_attention_mask = self.create_dec_cross_attention_mask(src = src)

            # [4] Decoder auto-regressive decoding loop
            dec_outputs = [BOS_IDX]
            for _ in range(max_len):
                # Convert token ID to tensor
                                        # dec_input_tensor: [1, dec_seq_len]
                dec_input_tensor = torch.tensor(dec_outputs).to(src.device).reshape(1, -1)

                # Create TRG mask for every iteration
                                        # trg_mask: [1, 1, dec_seq_len, dec_seq_len]
                dec_self_attention_mask = self.create_dec_self_attention_mask(trg = dec_input_tensor)

                # Decoder forward pass
                                        # decoder_out: [1, dec_seq_len, hidden_dim]
                decoder_out = self.decoder(
                    trg = dec_input_tensor,
                    encoder_out = encoder_out,
                    dec_self_attention_mask = dec_self_attention_mask,
                    dec_cross_attention_mask = dec_cross_attention_mask
                )

                # Output Layer forward pass (project decoder output to target vocabulary size)
                                        # out: [1, dec_seq_len, trg_vocab_size]
                out = self.output_layer(decoder_out)

                # Argmax to get the predicted token ID - Get the last token's prediction
                pred = out.argmax(dim=-1)[:, -1].item()  

                # Append predicted token ID to the decoder input for the next iteration
                dec_outputs.append(pred)

                # Stop if EOS token is generated
                if pred == EOS_IDX:
                    break

            return dec_outputs

    @staticmethod
    def create_enc_self_attention_mask(
        src: torch.Tensor
    ) -> torch.Tensor:
        batch_size, src_seq_len = src.shape
        
        # -- Remember -- 
        #     dot: [batch_size, num_heads, enc_q_len, enc_k_len]
        #     src_mask: [batch_size, 1, 1, enc_k_len] 
        # 
        # The encoder mask is a padding mask that indicates which tokens in the key
        # are padding tokens and should be ignored by the query during the dot product attention.

        # Create encoder mask (PADDING MASK) based on the source input
                                            # src: [batch_size, src_seq_len]
                                            # src_mask: [batch_size, 1, 1, enc_k_len]
        padding_mask = (src != PAD_IDX).view(batch_size, 1, 1, src_seq_len)

        return padding_mask

    @staticmethod
    def create_dec_self_attention_mask(
        trg: torch.Tensor
    ) -> torch.Tensor:
        batch_size, trg_seq_len = trg.shape

        # -- Remember --
        #     dot: [batch_size, num_heads, trg_q_len, trg_k_len]
        #     dec_mask: [batch_size, 1, trg_q_len, trg_k_len] 
        #
        #  The decoder mask is a combination of a padding mask and a causal mask.
        #     PADDING MASK: indicates which tokens in the key are padding tokens
        #     CAUSAL MASK: indicates which tokens in the key are future tokens 

        # Padding Mask
                                            # padding_mask: [batch_size, 1, 1, trg_k_len]
        padding_mask = (trg != PAD_IDX).view(batch_size, 1, 1, trg_seq_len)
        padding_mask = padding_mask.to(trg.device)

        # Causal Mask - construct lower triangle
                                            # causal_mask: [1, 1, trg_q_len, trg_k_len]
        tril_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len), device = trg.device)).bool()
        causal_mask = tril_mask.view(1, 1, trg_seq_len, trg_seq_len)

        # Combine Padding Mask and Causal Mask - Broadcasting magic
                                            # padding_mask: [batch_size, 1, 1, trg_k_len]
                                            # causal_mask: [1, 1, trg_q_len, trg_k_len]
                                            # dec_self_attention_mask: [batch_size, 1, trg_q_len, trg_k_len]
        dec_self_attention_mask = padding_mask & causal_mask

        return dec_self_attention_mask
        
    @staticmethod
    def create_dec_cross_attention_mask(
        src: torch.Tensor
    ) -> torch.Tensor:
        batch_size, src_seq_len = src.shape

        # The cross-attention mask is a padding mask that indicates which tokens in the encoder output key
        # are padding tokens and should be ignored by the decoder query during the dot product attention.
        # Implentation-wise, this is the same as encoder padding mask - create_enc_self_attention_mask()

        # Create cross-attention mask (PADDING MASK) based on the source input
                                            # src: [batch_size, src_seq_len]
                                            # cross_attention_mask: [batch_size, 1, 1, enc_k_len]
        cross_attention_mask = (src != PAD_IDX).view(batch_size, 1, 1, src_seq_len)
        cross_attention_mask = cross_attention_mask.to(src.device)

        return cross_attention_mask
        
class OutputLayer(nn.Module):
    def __init__(self,
        hidden_dim: int,
        trg_vocab_size: int
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(hidden_dim, trg_vocab_size)

    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
                                            # x: [batch_size, seq_len, hidden_dim]
                                            # out: [batch_size, seq_len, trg_vocab_size]
        x = self.linear(x)

        return x

################################################################################################################
#
#                                         Decoder and Decoder Layer 
#
################################################################################################################
class Decoder(nn.Module):
    def __init__(self,
        num_layers: int,
        trg_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_prob: float = 0.0,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim    # for scaling the token embedding

        # Vanilla Token Embedding Layer
        self.token_embedding_layer = TokenEmbeddingLayer(
            vocab_size = trg_vocab_size,
            embedding_dim = embedding_dim
        )

        # Positional Embedding Layer (Fixed Sinusoidal)
        self.position_embedding_layer = FixedPositionEmbeddingLayer(
            embedding_dim = embedding_dim,
            max_seq_len = max_seq_len
        )

        # Embedding Output Dropout Layer
        self.embedding_output_dropout_layer = nn.Dropout(dropout_prob)

        # Decoder Layers (Stacked)
        self.layers = nn.ModuleList([
            DecoderLayer(
                input_dim = embedding_dim if i == 0 else hidden_dim,
                hidden_dim = hidden_dim,
                num_heads = num_heads,
                ffn_dim = ffn_dim,
                dropout_prob = dropout_prob
            )
            for i in range(num_layers)
        ])

    def forward(self,
        trg: torch.Tensor, 
        encoder_out: torch.Tensor,
        dec_self_attention_mask: torch.Tensor | None = None,
        dec_cross_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Token Embedding
                                            # trg: [batch_size, seq_len]
                                            # token_embedding: [batch_size, seq_len, embedding_dim]
        token_embedding = self.token_embedding_layer(x = trg)

        # Position Embedding
                                            # position_embedding: [1, seq_len, embedding_dim]
        _, trg_seq_len = trg.shape
        position_embedding = self.position_embedding_layer(seq_len = trg_seq_len)

        # Token Embedding + Position Embedding + Dropout After Adding Positional Embedding
        #   in the bottom of page 5 of the paper, the token embedding layer is scaled by sqrt(d_model)
        #   before adding the position embedding,
                                            # y: [batch_size, seq_len, embedding_dim]
        y = (self.hidden_dim ** 0.5) * token_embedding + position_embedding
        y = self.embedding_output_dropout_layer(y)

        # Decoder Layers (Stacked) - Masked Self-Attention - Add & Norm - Cross-Attention - Add & Norm - FFN - Add & Norm
                                            # y: [batch_size, seq_len, hidden_dim]
                                            # encoder_out: [batch_size, enc_seq_len, hidden_dim]
                                            # dec_self_attention_mask: [batch_size, 1, trg_q_len, trg_k_len]  - Decoder mask (Combination of PADDING MASK and CAUSAL MASK)
                                            # dec_cross_attention_mask: [batch_size, 1, 1, enc_k_len]  - Cross-Attention mask (PADDING MASK)
        for layer in self.layers:
            y = layer(
                trg = y, 
                encoder_out = encoder_out, 
                dec_self_attention_mask = dec_self_attention_mask, 
                dec_cross_attention_mask = dec_cross_attention_mask
            )

        return y

class DecoderLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_prob: float = 0.0     
    ) -> None:
        super().__init__()

        # Decoder Self-Attention Layer 
        self.masked_attention_layer = MultiHeadScaledDotProductAttentionLayer(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout_prob = dropout_prob
        )

        # Add and Norm Layer 1
        self.add_and_norm_layer_1 = AddAndNormLayer(
            residual_dim = hidden_dim,
            dropout_prob = dropout_prob
        )

        # Decoder Cross-Attention Layer 
        self.attention_layer = MultiHeadScaledDotProductAttentionLayer(
            input_dim = hidden_dim,
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout_prob = dropout_prob
        )

        # Add and Norm Layer 2
        self.add_and_norm_layer_2 = AddAndNormLayer(
            residual_dim = hidden_dim,
            dropout_prob = dropout_prob
        )

        # Positionwise Feedforward Layer
        self.ffn_layer = FFNLayer(
            hidden_dim = hidden_dim,
            ffn_dim = ffn_dim,
            dropout_prob = dropout_prob
        )

        # Add and Norm Layer 3
        self.add_and_norm_layer_3 = AddAndNormLayer(
            residual_dim = hidden_dim,
            dropout_prob = dropout_prob
        )

    def forward(self,
        trg: torch.Tensor, 
        encoder_out: torch.Tensor,
        dec_self_attention_mask: torch.Tensor | None = None,
        dec_cross_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
                                            # trg: [batch_size, seq_len, hidden_dim]
                                            # dec_self_attention_mask: [batch_size, 1, seq_len, seq_len]  - Decoder mask (Combination of PADDING MASK and CAUSAL MASK)
                                            # dec_cross_attention_mask: [batch_size, 1, 1, enc_seq_len]  - Cross-Attention mask (PADDING MASK)
        # [1] Masked Self-attention (Decoder)
                                            # masked_attention_out: [batch_size, seq_len, hidden_dim]
        masked_attention_out = self.masked_attention_layer(
            query = trg,
            key = trg,
            value = trg,
            mask = dec_self_attention_mask
        )
        # [2] Add & Norm 1
                                            # enc_out: [batch_size, enc_seq_len, hidden_dim]
                                            # norm_out_1: [batch_size, seq_len, hidden_dim]
        norm_out_1 = self.add_and_norm_layer_1(
            identity = trg,
            prev_out = masked_attention_out
        )

        # [3] Cross-Attention with Encoder Output
                                            # dec_cross_attention_mask: [batch_size, 1, seq_len, enc_seq_len]
                                            # cross_attention_out: [batch_size, seq_len, hidden_dim]
        cross_attention_out = self.attention_layer(
            query = norm_out_1,
            key = encoder_out,
            value = encoder_out,
            mask = dec_cross_attention_mask
        )

        # [4] Add & Norm 2
                                            # norm_out_2: [batch_size, seq_len, hidden_dim]
        norm_out_2 = self.add_and_norm_layer_2(
            identity = norm_out_1,
            prev_out = cross_attention_out
        )

        # [5] Positionwise Feedforward
                                            # ffn_out: [batch_size, seq_len, hidden_dim]
        ffn_out = self.ffn_layer(norm_out_2)
        
        # [6] Add & Norm 3
                                            # norm_out_3: [batch_size, seq_len, hidden_dim]
        norm_out_3 = self.add_and_norm_layer_3(
            identity = norm_out_2,
            prev_out = ffn_out
        )

        return norm_out_3


################################################################################################################
#
#                                         Encoder and Encoder Layer 
#
################################################################################################################
class Encoder(nn.Module):
    def __init__(self,
        num_layers: int,
        src_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_prob: float = 0.0,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim # for scaling the token embedding

        # Vanilla Token Embedding Layer
        self.token_embedding_layer = TokenEmbeddingLayer(
            vocab_size = src_vocab_size,
            embedding_dim = embedding_dim
        )

        # Positional Embedding Layer (Fixed Sinusoidal)
        self.position_embedding_layer = FixedPositionEmbeddingLayer(
            embedding_dim = embedding_dim,
            max_seq_len = max_seq_len
        )

        # Embedding Output Dropout Layer
        self.embedding_output_dropout_layer = nn.Dropout(dropout_prob)

        # Encoder Layers (Stacked)
        self.layers = nn.ModuleList([
            EncoderLayer(
                input_dim = embedding_dim if i == 0 else hidden_dim,
                hidden_dim = hidden_dim,
                num_heads = num_heads,
                ffn_dim = ffn_dim,
                dropout_prob = dropout_prob
            )
            for i in range(num_layers)
        ])

    def forward(self, 
        src: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Token Embedding
                                            # src: [batch_size, seq_len]
                                            # token_embedding: [batch_size, seq_len, embedding_dim]
        token_embedding = self.token_embedding_layer(x = src)

        # Position Embedding
                                            # position_embedding: [1, seq_len, embedding_dim]
        _, src_seq_len = src.shape
        position_embedding = self.position_embedding_layer(seq_len = src_seq_len)

        # Token Embedding + Position Embedding + Dropout After Adding Positional Embedding
        #   in the bottom of page 5 of the paper, the token embedding layer is scaled by sqrt(d_model)
        #   before adding the position embedding,
                                            # x: [batch_size, seq_len, embedding_dim]
        x = (self.hidden_dim ** 0.5) * token_embedding + position_embedding
        x = self.embedding_output_dropout_layer(x)

        # Encoder Layers (Stacked) - Attention - Add & Norm - FFN - Add & Norm
                                            # x: [batch_size, seq_len, hidden_dim]
                                            # mask: [batch_size, 1, 1, seq_len]  - Encoder mask (PADDING MASK)
        for layer in self.layers:
            x = layer(
                src = x, 
                mask = mask
            )

        return x

class EncoderLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_prob: float = 0.0     
    ) -> None:
        super().__init__()

        # Self-Attention Layer
        self.attention_layer = MultiHeadScaledDotProductAttentionLayer(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            dropout_prob = dropout_prob
        )

        # Add and Norm Layer 1
        self.add_and_norm_layer_1 = AddAndNormLayer(
            residual_dim = hidden_dim,
            dropout_prob = dropout_prob
        )

        # Positionwise Feedforward Layer
        self.ffn_layer = FFNLayer(
            hidden_dim = hidden_dim,
            ffn_dim = ffn_dim,
            dropout_prob = dropout_prob
        )

        # Add and Norm Layer 2
        self.add_and_norm_layer_2 = AddAndNormLayer(
            residual_dim = hidden_dim,
            dropout_prob = dropout_prob
        )

    def forward(self, 
        src: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
                                            # x: [batch_size, seq_len, hidden_dim]
                                            # mask: [batch_size, 1, 1, seq_len]  - Encoder mask (PADDING MASK)
        # [1] Self-attention
                                            # attention_out: [batch_size, seq_len, hidden_dim]
        attention_out = self.attention_layer(
            query = src,
            key = src,
            value = src,
            mask = mask
        )
        # [2] Add & Norm
                                            # norm_out_1: [batch_size, seq_len, hidden_dim]
        norm_out_1 = self.add_and_norm_layer_1(
            identity = src,
            prev_out = attention_out
        )

        # [3] Positionwise Feedforward
                                            # ffn_out: [batch_size, seq_len, hidden_dim]
        ffn_out = self.ffn_layer(norm_out_1)
        
        # [4] Add & Norm
                                            # norm_out_2: [batch_size, seq_len, hidden_dim]
        norm_out_2 = self.add_and_norm_layer_2(
            identity = norm_out_1,
            prev_out = ffn_out
        )

        return norm_out_2


################################################################################################################
#
#                             Attention Layer, Feedforward Layer, Add & Norm Layer
#
################################################################################################################
class MultiHeadScaledDotProductAttentionLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_prob: float = 0.0 
    ) -> None: 
        
        super().__init__()

        assert hidden_dim % num_heads == 0, 'hidden_dim should be a integer multiple of num_heads'

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.head_dim = hidden_dim // num_heads

        self.fc_q = nn.Linear(input_dim, hidden_dim)
        self.fc_k = nn.Linear(input_dim, hidden_dim)
        self.fc_v = nn.Linear(input_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p = self.dropout_prob)
        self.scale = self.head_dim ** 0.5

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = query.shape[0]
                                            # query: [batch_size, q_len, input_dim]
                                            # key: [batch_size, k_len, input_dim]
                                            # value: [batch_size, v_len, input_dim]
                                            # 
                                            # Separating the linear layers for query, key, value allows us 
                                            # to have different projections for each of them, which can be 
                                            # beneficial for learning different representations for queries, keys, and values.
                                            # 
                                            # In encoder self-attention, query = key = value = x (input to the encoder layer)
                                            # In decoder self-attention, query = key = value = y (input to the decoder layer)
                                            # In decoder cross-attention, query = y (input to the decoder layer), 
                                            #                             key = value = encoder_output
        # Feedforward
                                            # q: [batch_size, q_len, hidden_dim]
                                            # k: [batch_size, k_len, hidden_dim]
                                            # v: [batch_size, v_len, hidden_dim]
                                            # 
                                            # A better way to do this is to combine 
                                            # the three linear layers into one linear layer 
                                            # that outputs 3*hidden_dim, 
                                            # and then chunk the output into q, k, v. 
                                            # This is more efficient because it reduces the number 
                                            # of sequential matrix multiplications from 3 to 1. 
                                            # 
                                            # However, for clarity and simplicity, 
                                            # we will keep them separate in this implementation.
        q, k, v = self.fc_q(query), self.fc_k(key), self.fc_v(value)

        # Chunk [hidden_dim] to [num_heads, head_dim] and re-arrange
                                            # original: [batch_size, seq_len, hidden_dim]
                                            # chunked: [batch_size, seq_len, num_heads, head_dim]
                                            # re-arranged: [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute the attention score per head (un-normalized attention)
                                            # q: [batch_size, num_heads, q_len, head_dim]
                                            # k.T: [batch_size, num_heads, head_dim, k_len]
                                            # dot: [batch_size, num_heads, q_len, k_len]
        dot = q @ k.permute(0, 1, 3, 2)
        scaled_dot = dot / self.scale

        # [Optional] Apply (PADDING or CAUSAL) mask
                                            # mask: [batch_size, 1, 1, k_len]       if encoder (PADDING MASK)
                                            # mask: [batch_size, 1, q_len, k_len]   if decoder (CAUSAL MASK)
        if mask is not None:
            scaled_dot = scaled_dot.masked_fill(~mask, float('-inf'))

        # Compute attention weight/distribution per-head
                                            # scaled_dot: [batch_size, num_heads, q_len, k_len]
                                            # exp_scaled_dot: [batch_size, num_heads, q_len, k_len]
                                            # sum_exp_scaled_dot: [batch_size, num_heads, q_len, 1]

                                            # attention_weight = exp_scaled_dot / sum_exp_scaled_dot
                                            # attention_weight: [batch_size, num_heads, q_len, k_len]
        attention_weight = scaled_dot.softmax(dim=-1)

        # [Optional] Apply dropout to attention distribution
        attention_weight = self.dropout(attention_weight)

        # Compute sum of value weighted by attention_weight
                                            # attention_weight: [batch_size, num_heads, q_len, k_len]
                                            # v: [batch_size, num_heads, v_len, head_dim]
                                            # weighted_value: [batch_size, num_heads, q_len, head_dim]
        weighted_value = attention_weight @ v

        # Re-arrange again (Collapse multiple heads to single head)
                                            # weighted_value: [batch_size, num_heads, q_len, head_dim]
                                            # attention_output: [batch_size, q_len, num_heads * head_dim]
                                            # 
                                            # Use .contiguous() after permute() or transpose()
                                            # to ensure the tensor is stored in a contiguous chunk of memory,
                                            # which is required for .view() to work correctly.
                                            #
                                            # Or just use .reshape() instead of .view(), which automatically
                                            # handles non-contiguous tensors.
        weighted_value = weighted_value.permute(0, 2, 1, 3)
        attention_output = weighted_value.reshape(batch_size, -1, self.hidden_dim)

        # Final Feedforward mixing
        output = self.fc_o(attention_output)

        return output

class FFNLayer(nn.Module):
    def __init__(self,
        hidden_dim: int,
        ffn_dim: int,
        dropout_prob: float = 0.0
    ) -> None:
        super().__init__()

        self.ff1 = nn.Linear(hidden_dim, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Feedforward
                                            # x: [batch_size, seq_len, hidden_dim]
                                            # ff1_out: [batch_size, seq_len, ffn_dim]
                                            # ff2_out: [batch_size, seq_len, hidden_dim]
        ff1_out = self.dropout(torch.relu(self.ff1(x)))
        ff2_out = self.dropout(self.ff2(ff1_out))

        return ff2_out

class AddAndNormLayer(nn.Module):
    def __init__(self, 
        residual_dim: int,
        dropout_prob: float = 0.0
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(residual_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
        identity: torch.Tensor,
        prev_out: torch.Tensor
    ) -> torch.Tensor:
                                            
        out = self.layer_norm(identity + self.dropout(prev_out))

        return out 

################################################################################################################
#
#                  Token Embedding Layer, Position Embedding Layer, Positional Encoding Layer
#
################################################################################################################
class TokenEmbeddingLayer(nn.Module):
    def __init__(self,
        vocab_size: int,
        embedding_dim: int
    ) -> None:
        super().__init__()

        # Embedding layer - also set the padding idx
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size, 
            embedding_dim = embedding_dim,
            padding_idx = PAD_IDX
        )

    def forward(self,
        x: torch.Tensor | torch.LongTensor
    ) -> torch.Tensor:
                                            # x: [batch_size, seq_len]
                                            # out: [batch_size, seq_len, embedding_dim]
        x = x.long()
        out = self.token_embedding(x)

        return out

# Generate Position Embeddings for a given sequence length and embedding dimension
# Fixed sinusoidal position embedding as described in the original Transformer paper (Vaswani et al., 2017)
class FixedPositionEmbeddingLayer(nn.Module):
    def __init__(self,
        embedding_dim: int,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Precompute the position embeddings for the maximum sequence length
        pos = torch.arange(max_seq_len, dtype = torch.float)      # [max_seq_len]

        # Compute denominator
        two_i_len = embedding_dim // 2 + embedding_dim % 2      
        two_i = torch.arange(two_i_len, dtype = torch.float) * 2  # [embedding_dim // 2 + embedding_dim % 2]
        denominator = torch.exp(                                # [embedding_dim // 2 + embedding_dim % 2]
            two_i / embedding_dim * torch.log(torch.FloatTensor([10000.0]))
        )  

        # Sinusoidal input broadcasting
                                                                # pos: [max_seq_len, 1]
                                                                # denominator: [1, embedding_dim // 2 + embedding_dim % 2]
                                                                # sinu_input: [max_seq_len, embedding_dim // 2 + embedding_dim % 2]
        sinu_input = pos.view(-1, 1) / denominator.view(1, -1)

        # Precompute the position embeddings
        pe = torch.zeros(max_seq_len, embedding_dim)            # [max_seq_len, embedding_dim]
                                                                # sinu_input: [max_seq_len, embedding_dim // 2 + embedding_dim % 2]
                                                                # Even indices: [max_seq_len, embedding_dim // 2 + embedding_dim % 2]
                                                                # Odd indices: [max_seq_len, embedding_dim // 2]
        pe[:, 0::2] = torch.sin(sinu_input)            
        pe[:, 1::2] = torch.cos(sinu_input[:, :embedding_dim//2])     

        # Store the precomputed position embeddings for tensor slicing during the forward pass
        # Storing as register_buffer means it will be saved and moved to the appropriate device with the model, 
        # but it will not be updated during training (i.e., it's not a learnable parameter).
                                                                # position_embeddings: [1, max_seq_len, embedding_dim]
        self.register_buffer('position_embeddings', pe.view(1, max_seq_len, embedding_dim))     
        
    def forward(self,
        seq_len: int = None
    ) -> torch.Tensor:
        
        if seq_len is None:
            seq_len = self.max_seq_len
        elif seq_len > self.max_seq_len:
            raise ValueError(f'seq_len should be less than or equal to max_seq_len ({self.max_seq_len})')

        # Slice the precomputed position embeddings to match the input sequence length
                                            # out: [1, seq_len, embedding_dim]
        out = self.position_embeddings[:, :seq_len, :]

        return out

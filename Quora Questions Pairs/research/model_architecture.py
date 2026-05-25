import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    MODEL_TYPE = "LSTM_attention"
    ATTENTION_TYPE = "ESIM_Style-MultiHead-CrossAttention-Bahdanau"

    NUM_HEADS = 4                                     
    SELF_ATTENTION_NUM_HEADS = NUM_HEADS
    CROSS_ATTENTION_NUM_HEADS = NUM_HEADS

    ATTENTION_DROPOUT = 0.0
    SKIP_CONNECTION_IN_ATTENTION = True
    POOLING_TYPE = "Average and Max"
    
    # Embedding
    EMB_NORM = False
    FREEZE_TOKEN_EMBEDDING = True
    TOKEN_EMBEDDING = "gloVe-6B-100d"
    EMB_DIM = 100
    EMB_DP = 0.0

    # Model
    LOSS = "BCE with Logits"
    ENHC_LSTM_DROPOUT = 0.4
    ENHC_LSTM_NUM_LAYERS = 2
    ENHC_LSTM_DIM = 256
    ENHC_LSTM_BIDIRECTIONAL = True
    ENHC_LSTM_OUT = ENHC_LSTM_DIM * (2 if ENHC_LSTM_BIDIRECTIONAL else 1)
    ENHC_LSTM_NORM = False
    COMP_LSTM_DIM = 128
    COMP_LSTM_BIDIRECTIONAL = True
    COMP_LSTM_OUT = COMP_LSTM_DIM * (2 if COMP_LSTM_BIDIRECTIONAL else 1)
    COMP_LSTM_NORM = False
    COMP_LSTM_DROPOUT = 0.3
    COMP_LSTM_NUM_LAYERS = 2
    PROJ_DIMS = [256]
    PROJ_DROPOUT = 0.3

    # Loss specific settings
    if LOSS == "Contrastive Loss":
        MARGIN = 1.0
    elif LOSS == "BCE with Logits":
        LABEL_SMOOTHING = 0.05
        FC_DIMS = [1024, 512]
        FC_DP = 0.5

    MASK_FILL_NUM = -1e10

    @classmethod
    def to_dict(cls):
        return {
            k.lower(): v for k, v in cls.__dict__.items()
            if not k.startswith("_")
            and not inspect.isroutine(v)
            and not isinstance(v, (classmethod, staticmethod))
        }

class CrossAttentionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim, mask_fill_num=model_cfg.MASK_FILL_NUM,
                 dropout=model_cfg.ATTENTION_DROPOUT):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, proj_dim)
        self.W_k = nn.Linear(hidden_dim, proj_dim)
        self.W_v = nn.Linear(hidden_dim, proj_dim)
        self.V = nn.Linear(proj_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.mask_fill_num = mask_fill_num

    def forward(self, query, key_value, mask_kv):
        Q = self.W_q(query)
        K = self.W_k(key_value)
        V = self.W_v(key_value)
        energy = torch.tanh(Q.unsqueeze(2) + K.unsqueeze(1))
        scores = self.V(energy).squeeze(-1)
        scores = scores.masked_fill(mask_kv.unsqueeze(1) == 0, self.mask_fill_num)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        aligned = torch.bmm(attn_weights, V)
        return aligned

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=model_cfg.CROSS_ATTENTION_NUM_HEADS):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            CrossAttentionHead(hidden_dim, head_dim) for _ in range(num_heads)
        ])
        self.out_linear = nn.Linear(head_dim*num_heads, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, query, key_value, mask_kv):
        x = torch.cat([h(query, key_value, mask_kv) for h in self.heads], dim=-1)
        x = self.out_linear(x)
        return self.norm(query + x)

class AvgMaxPool(nn.Module):
    def __init__(self, mask_fill_num=model_cfg.MASK_FILL_NUM):
        super().__init__()
        self.mask_fill_num = mask_fill_num

    def forward(self, x, mask):
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_pool = (x * mask.unsqueeze(-1).sum(dim=1)) / lengths

        x_masked = mask.masked_fill(mask == 0, self.mask_fill_num)
        max_pool, _ = x_masked.max(dim=1)

        return torch.cat([avg_pool, max_pool], dim=-1)

class QuoraSiameseClassifier(nn.Module):
    def __init__(self, vocab_size, config=model_cfg, embedding=None, stop_mask=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.EMB_DIM)
        self.emb_norm = nn.LayerNorm(config.EMB_DIM) if config.EMB_NORM else nn.Identity()
        self.emb_dropout = nn.Dropout(config.EMB_DP)
        if stop_mask is not None:
            self.register_buffer("stop_mask", stop_mask)
        else:
            self.stop_mask = None
        if embedding is not None:
            print("Glove copied in Embedding Layer...")
            self.embedding.weight.data.copy_(embedding)
            self.embedding.weight.requires_grad = not config.FREEZE_TOKEN_EMBEDDING

        self.enhc_lstm = nn.LSTM(
            input_size=config.EMB_DIM,
            hidden_size=config.ENHC_LSTM_DIM,
            bidirectional=config.ENHC_LSTM_BIDIRECTIONAL,
            num_layers=config.ENHC_LSTM_NUM_LAYERS,
            dropout=config.ENHC_LSTM_DROPOUT if config.ENHC_LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True
        )
        self.enhc_lstm_norm = nn.LayerNorm(config.ENHC_LSTM_OUT) if config.ENHC_LSTM_NORM else nn.Identity()
        self.cross_attention = MultiHeadCrossAttention(config.ENHC_LSTM_OUT)
        self.proj = self._build_fc_layers(
            input_dim=4*config.ENHC_LSTM_OUT,
            fc_dims=config.PROJ_DIMS,
            dropout=config.PROJ_DROPOUT
        )
        self.comp_lstm = nn.LSTM(
            input_size=config.PROJ_DIMS[-1],
            hidden_size=config.COMP_LSTM_DIM,
            bidirectional=config.COMP_LSTM_BIDIRECTIONAL,
            num_layers=config.COMP_LSTM_NUM_LAYERS,
            dropout=config.COMP_LSTM_DROPOUT if config.COMP_LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True
        )
        self.comp_lstm_norm = nn.LayerNorm(config.COMP_LSTM_OUT) if config.COMP_LSTM_NORM else nn.Identity()
        self.pool = AvgMaxPool(mask_fill_num=model_cfg.MASK_FILL_NUM)
        self.fc_dims = self._build_fc_layers(
            input_dim=config.COMP_LSTM_OUT,
            fc_dims=config.FC_DIMS,
            dropout=config.FC_DP
        )
        self.final_layer = nn.Linear(config.FC_DIMS[-1], 1)
        
    def _build_fc_layers(self, input_dim, fc_dims, dropout):
        layers = []
        for dim in fc_dims:
            layers += [
                nn.Linear(input_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            input_dim = dim
        return nn.Sequential(*layers)
    
    def _create_mask(self, question):
        return (question != 0).float()

    def _encode(self, question):
        emb = self.embedding(question)
        emb = self.emb_norm(emb)
        emb = self.emb_dropout(emb)
        mask = self._create_mask(question)
        if self.stop_mask is not None:
            token_stop_mask = self.stop_mask[question]
            mask = mask * token_stop_mask.float()
        out, _ = self.enhc_lstm(emb)
        out = self.enhc_lstm_norm(out)
        return out, mask

    def forward(self, q1, q2):
        out1, mask1 = self._encode(q1)
        out2, mask2 = self._encode(q2)
        
        cross1 = self.cross_attention(query=out1, key_value=out2, mask_kv=mask2)
        cross2 = self.cross_attention(query=out2, key_value=out1, mask_kv=mask1)

        enhc1 = torch.cat([out1, cross1, out1 - cross1, out1*cross1], dim=-1)
        enhc2 = torch.cat([out2, cross2, out2 - cross2, out2*cross2], dim=-1)

        proj1 = self.proj(enhc1)
        proj2 = self.proj(enhc2)

        comp1, _ = self.comp_lstm(proj1)
        comp2, _ = self.comp_lstm(proj2)
        comp1 = self.comp_lstm_norm(comp1)
        comp2 = self.comp_lstm_norm(comp2)
        
        h1 = self.pool(comp1, mask1)
        h2 = self.pool(comp2, mask2)
        feat = torch.cat([h1, h2], dim=-1)
        final_feat = self.fc_dims(feat)
        logits = self.final_layer(final_feat)
        
        return logits.squeeze(-1)
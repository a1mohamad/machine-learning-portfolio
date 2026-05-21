import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    MODEL_TYPE = "LSTM_attention"
    ATTENTION_TYPE = "MultiHead-CrossAttention-Bahdanau"
    CNN_PLACE = "Before-LSTM"
    USE_SELF_ATTENTION = True

    NUM_HEADS = 4                                     
    SELF_ATTENTION_NUM_HEADS = NUM_HEADS
    CROSS_ATTENTION_NUM_HEADS = NUM_HEADS

    ATTENTION_DROPOUT = 0.0
    POOLING_TYPE = "MaskedMean"

    CNN_KERNEL_SIZES = [2, 3]
    CNN_DROPOUT = 0.1
    
    # Embedding
    LAYER_NORM_EMB = False
    FREEZE_TOKEN_EMBEDDING = True
    TOKEN_EMBEDDING = "gloVe-6B-100d"
    EMB_DIM = 100
    EMB_DP = 0.0

    # Model
    LOSS = "BCE with Logits"
    BIDIRECTIONAL = True
    DROPOUT = 0.4
    HIDDEN_DIM = 256
    LSTM_OUT = HIDDEN_DIM * (2 if BIDIRECTIONAL else 1)

    LAYER_NORM_LSTM = False
    ATTENTION_PROJECTION = False
    if ATTENTION_PROJECTION:
        PROJECT_DIM = HIDDEN_DIM // 2
    ENC_DIM = PROJECT_DIM if ATTENTION_PROJECTION else LSTM_OUT

    # Loss specific settings
    if LOSS == "Contrastive Loss":
        MARGIN = 1.0
    elif LOSS == "BCE with Logits":
        LABEL_SMOOTHING = 0.05
        FC_DIMS = [1024, 512]
        FC_DP = 0.5
        SIAMESE_SIMILARITY_PARAM = [
            "Encoded Q1",
            "Encoded Q2",
            "Multiplication Q1, Q2",
            "Abs Subtract Q1, Q2",
            "Cosine Similarity",
        ]
        MULTIPLE_FC_PARAM = sum(
            1 for param in SIAMESE_SIMILARITY_PARAM
            if "Q1" in param or "Q2" in param
        )
        INPUT_FC_DIM = MULTIPLE_FC_PARAM * ENC_DIM
        if any("Cosine" in param for param in SIAMESE_SIMILARITY_PARAM):
            INPUT_FC_DIM += 1

    MASK_FILL_NUM = -1e10
    NUM_LAYERS = 2
    SKIP_CONNECTION = False

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

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k, padding='same')
            for k in kernels
        ])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(out_channels*len(kernels), out_channels, kernel_size=1)

    def forward(self, x):
        convs_out = [self.dropout(self.activation(conv(x))) for conv in self.convs]
        x = torch.cat(convs_out, dim=1)
        x = self.proj(x)
        return x

class MaskedMeanPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask):
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths
        return pooled

class SelfAttentivePooling(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.attention = SelfAttentionHead(hidden_dim, proj_dim)
        
    def forward(self, x, mask):
        context = self.attention(x, mask)
        return context

class QuoraSiameseClassifier(nn.Module):
    def __init__(self, vocab_size, config=model_cfg, embedding=None, stop_mask=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.EMB_DIM)
        self.emb_norm = nn.LayerNorm(config.EMB_DIM)
        self.emb_dropout = nn.Dropout(config.EMB_DP)
        if stop_mask is not None:
            self.register_buffer("stop_mask", stop_mask)
        else:
            self.stop_mask = None
        if embedding is not None:
            print("Glove copied in Embedding Layer...")
            self.embedding.weight.data.copy_(embedding)
            self.embedding.weight.requires_grad = not config.FREEZE_TOKEN_EMBEDDING

        self.LSTM = nn.LSTM(
            input_size=config.EMB_DIM,
            hidden_size=config.HIDDEN_DIM,
            bidirectional=config.BIDIRECTIONAL,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0,
            batch_first=True
        )
        self.cnn = CNN(
            in_channels=model_cfg.EMB_DIM,
            out_channels=model_cfg.EMB_DIM,
            kernels=model_cfg.CNN_KERNEL_SIZES,
            dropout=model_cfg.CNN_DROPOUT
        )
        self.lstm_norm = nn.LayerNorm(config.LSTM_OUT)
        self.cross_attention = MultiHeadCrossAttention(config.LSTM_OUT)
        if config.POOLING_TYPE == "MaskedMean":
            self.pool = MaskedMeanPool()
        elif config.POOLING_TYPE == "Self-Additive":
            self.pool = SelfAttentivePooling(config.LSTM_OUT, config.LSTM_OUT)
        else:
            ValueError(f"Unknown pooling type: {config.POOLING_TYPE}")
        
        if config.ATTENTION_PROJECTION:
            self.proj = nn.Linear(config.LSTM_OUT, config.PROJECT_DIM)
        else:
            self.proj = nn.Identity()
        
        self.fc_dims = self._build_fc_layers(
            input_dim=config.INPUT_FC_DIM,
            fc_dims=config.FC_DIMS,
            dropout=config.FC_DP
        )

    def _build_fc_layers(self, input_dim, fc_dims, dropout):
        layers = []
        for dim in fc_dims:
            layers += [
                nn.Linear(input_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))   # final logit projection
        return nn.Sequential(*layers)
    
    def _create_mask(self, question):
        return (question != 0).float()

    def _encode(self, question):
        emb = self.embedding(question)
        if self.config.LAYER_NORM_EMB:
            emb = self.emb_norm(emb)
        emb = self.emb_dropout(emb)
        mask = self._create_mask(question)
        if self.stop_mask is not None:
            token_stop_mask = self.stop_mask[question]
            mask = mask * token_stop_mask.float()
        emb = emb.transpose(1, 2)
        emb = self.cnn(emb)
        emb = emb.transpose(1, 2)
        out, _ = self.LSTM(emb)
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
        return out, mask

    def forward(self, q1, q2):
        out1, mask1 = self._encode(q1)
        out2, mask2 = self._encode(q2)
        cross1 = self.cross_attention(query=out1, key_value=out2, mask_kv=mask2)
        cross2 = self.cross_attention(query=out2, key_value=out1, mask_kv=mask1)
        h1 = self.pool(cross1, mask1)
        h2 = self.pool(cross2, mask2)
        h1, h2 = self.proj(h1), self.proj(h2)
        cosine_sim = F.cosine_similarity(h1, h2).unsqueeze(-1)
        feat = torch.cat([h1, h2, abs(h1 - h2), h1*h2, cosine_sim], dim=1)
        logits = self.fc_dims(feat)
        
        return logits.squeeze(-1)
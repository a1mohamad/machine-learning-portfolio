import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    MODEL_TYPE = "LSTM_attention"
    ATTENTION_TYPE = "MultiHead-Bahdanau"
    # Embedding
    LAYER_NORM_EMB = False
    FREEZE_TOKEN_EMBEDDING = True
    TOKEN_EMBEDDING = "gloVe-6B-100d"
    EMB_DIM = 100
    EMB_DP = 0.0
    # Model
    LOSS = "BCE with Logits"
    NUM_HEADS = 4
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    HIDDEN_DIM = 512
    LSTM_OUT = HIDDEN_DIM*(2 if BIDIRECTIONAL else 1)
    ATTENTION_DROPOUT = 0.0
    LAYER_NORM_LSTM = True
    LAYER_NORM_ATTENTION = False
    ATTENTION_PROJECTION = False
    if ATTENTION_PROJECTION:
        PROJECT_DIM = HIDDEN_DIM // 2
    ENC_DIM = PROJECT_DIM if ATTENTION_PROJECTION else LSTM_OUT
    if LOSS == "Contrastive Loss":
        MARGIN = 1.0
    elif LOSS == "BCE with Logits":
        LABEL_SMOOTHING = 0.05
        FC_DIMS = [1024, 512]
        FC_DP = 0.5
        SIAMESE_SIMILARITY_PARM = ["Encoded Q1", "Encoded Q2", "Multiplication Q1, Q2", "Abs Subtract Q1, Q2", "Cosine Similarity"]
        MULTIPLE_FC_PARAM = sum(1 for param in SIAMESE_SIMILARITY_PARM
                     if "Q1" in param or "Q2" in param)
        INPUT_FC_DIM = MULTIPLE_FC_PARAM * ENC_DIM
        if any("Cosine" in param for param in SIAMESE_SIMILARITY_PARM):
            INPUT_FC_DIM += 1
    MASK_FILL_NUM = -1e10
    NUM_LAYERS = 2
    SKIP_CONNECTION = False
    @classmethod
    def to_dict(cls):
        return {
            k.lower(): v for k, v in cls.__dict__.items()
            if not k.startswith("_")
            and not inspect.isroutine(v)   # functions, methods
            and not isinstance(v, (classmethod, staticmethod))
        }

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim, mask_fill_num=model_cfg.MASK_FILL_NUM,
                 dropout=model_cfg.ATTENTION_DROPOUT):
        super().__init__()
        self.W = nn.Linear(hidden_dim, proj_dim)           # project to subspace
        self.V = nn.Linear(proj_dim, 1, bias=False)        # score
        self.mask_fill_num = mask_fill_num
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, mask):
        proj = self.W(lstm_output)                         # [B, L, proj_dim]
        energy = torch.tanh(proj)
        scores = self.V(energy).squeeze(-1)                # [B, L]
        scores = scores.masked_fill(mask == 0, self.mask_fill_num)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Pool from the projected features (not original lstm_output)
        masked_proj = proj * mask.unsqueeze(-1)
        context = torch.bmm(weights.unsqueeze(1), masked_proj).squeeze(1)   # [B, proj_dim]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=model_cfg.NUM_HEADS):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(hidden_dim, head_dim) for _ in range(num_heads)
        ])
        self.out_linear = nn.Linear(head_dim*num_heads, hidden_dim)

    def forward(self, hidden_state, mask):
        x = torch.cat([h(hidden_state, mask) for h in self.heads], dim=-1)
        x = self.out_linear(x)
        return x

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
        self.lstm_norm = nn.LayerNorm(config.LSTM_OUT)
        self.attention = MultiHeadAttention(config.LSTM_OUT)
        if config.ATTENTION_PROJECTION:
            self.proj = nn.Linear(config.LSTM_OUT, config.PROJECT_DIM)
        else:
            self.proj = nn.Identity()
        
        self.attn_norm = nn.LayerNorm(config.LSTM_OUT)
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
        out, _ = self.LSTM(emb)
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
        ctx = self.attention(out, mask)
        if self.config.LAYER_NORM_ATTENTION:
            ctx = self.attn_norm(ctx)
        return ctx

    def forward(self, q1, q2):
        h1 = self._encode(q1)
        h2 = self._encode(q2)
        h1, h2 = self.proj(h1), self.proj(h2)
        cosine_sim = F.cosine_similarity(h1, h2).unsqueeze(-1)
        feat = torch.cat([h1, h2, abs(h1 - h2), h1*h2, cosine_sim], dim=1)
        logits = self.fc_dims(feat)
        
        return logits.squeeze(-1)

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
        lstm_output_dim = config.HIDDEN_DIM*(2 if config.BIDIRECTIONAL else 1)
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        self.attention = MultiHeadAttention(lstm_output_dim)
        if config.ATTENTION_PROJECTION:
            self.proj = nn.Linear(lstm_output_dim, config.PROJECT_DIM)
        else:
            self.proj = nn.Identity()
        self.attn_norm = nn.LayerNorm(lstm_output_dim)

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
        out, _ = self.LSTM(emb)
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
        ctx = self.attention(out, mask)
        if self.config.LAYER_NORM_ATTENTION:
            ctx = self.attn_norm(ctx)
        return ctx

    def _encode_with_attention(self, question):
        emb = self.embedding(question)
        if self.config.LAYER_NORM_EMB:
            emb = self.emb_norm(emb)
        emb = self.emb_dropout(emb)
        mask = self._create_mask(question)
        if self.stop_mask is not None:
            token_stop_mask = self.stop_mask[question]
            mask = mask * token_stop_mask.float()
    
        out, _ = self.LSTM(emb)
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
    
        # Get context and attention weights
        ctx, attn_weights = self.attention(out, mask, return_weights=True)   # attn_weights: list of [B, L]
    
        if self.config.LAYER_NORM_ATTENTION:
            ctx = self.attn_norm(ctx)
    
        # Project (or identity)
        final = self.proj(ctx)
    
        return final, attn_weights, mask

    def forward(self, q1, q2):
        h1 = self._encode(q1)
        h2 = self._encode(q2)
        h1, h2 = self.proj(h1), self.proj(h2) 
        dist = F.pairwise_distance(h1, h2, p=2)
        
        return dist
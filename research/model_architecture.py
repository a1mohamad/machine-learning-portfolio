import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    MODEL_TYPE = "LSTM-attention"
    ATTENTION_TYPE = "OneHead-Bahdanau"
    # Embedding
    LAYER_NORM_EMB = True
    FREEZE_TOKEN_EMBEDDING = True
    TOKEN_EMBEDDING = "gloVe-6B-100d"
    EMB_DIM = 100
    EMB_DP = 0.2
    # Model
    LOSS = "Consrative Loss"
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    HIDDEN_DIM = 256
    ATTENTION_DROPOUT = 0.2
    LAYER_NORM_LSTM = False
    LAYER_NORM_ATTENTION = True
    ATTENTION_PROJECTION = False
    if ATTENTION_PROJECTION:
        PROJECT_DIM = HIDDEN_DIM // 2
    MARGIN = 2.0
    MASK_FILL_NUM = -1e10
    SIAMESE_SIMILARITY_PARM = ["Euclidean Distance"]
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

class Attention(nn.Module):
    def __init__(self, hidden_dim, mask_fill_num=model_cfg.MASK_FILL_NUM, dropout=model_cfg.ATTENTION_DROPOUT):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
        self.mask_fill_num = mask_fill_num
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_output, mask):
        energy = torch.tanh(self.W(lstm_output))
        scores = self.V(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, self.mask_fill_num) 
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        lstm_masked = lstm_output * mask.unsqueeze(-1)
        weights = weights.unsqueeze(1)
        context = torch.bmm(weights, lstm_masked)
        
        return context.squeeze(1), weights

class QuoraSiameseClassifier(nn.Module):
    def __init__(self, vocab_size, config=model_cfg, embedding=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.EMB_DIM)
        self.emb_norm = nn.LayerNorm(config.EMB_DIM)
        self.emb_dropout = nn.Dropout(config.EMB_DP)
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
        self.attention = Attention(lstm_output_dim)
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
        out, _ = self.LSTM(emb)
        if self.config.LAYER_NORM_LSTM:
            out = self.lstm_norm(out)
        ctx, _ = self.attention(out, mask)
        if self.config.LAYER_NORM_ATTENTION:
            ctx = self.attn_norm(ctx)
        return ctx

    def forward(self, q1, q2):
        h1 = self.proj(self._encode(q1))
        h2 = self.proj(self._encode(q2))
        return h1, h2
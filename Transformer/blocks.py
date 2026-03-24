import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Adding special tokens:
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.word_to_id["<BOS>"] = 2
        self.word_to_id["<EOS>"] = 3
        self.vocab_size += 4

        for sentence in texts:
            for word in sentence.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.split()
        ids: List[int] = []
        for word in words:
            if word not in self.word_to_id:
                ids.append(1)
            else:
                ids.append(self.word_to_id[word])

        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word[id] for id in ids]
        return " ".join(words)


class PositionalEncoding(nn.Module):
    """
    Create a Positional Encoding Layer
    """

    def __init__(self):
        super().__init__()

    # Recieve a 2d matrix
    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        self.num_of_tokens = input_tensor.shape[1]
        self.d_model = input_tensor.shape[2]
        vector1 = torch.arange(self.num_of_tokens).reshape(-1, 1).float()
        # Vector from 0 to n-1 Shape(n, 1)

        vector2 = torch.arange(self.d_model/2).reshape(-1, 1)
        vector3 = torch.e**(-2*vector2*math.log(10000)/self.d_model).T
        # Vector for division terms Shape(1, d_model)

        matrix = torch.matmul(vector1, vector3)
        # Matrix with shape(num_of_tokens, d_model/2)

        cosine_matrix = torch.cos(matrix)
        sine_matrix = torch.sin(matrix)
        
        # 1. Create an empty matrix of shape (num_of_tokens, d_model)
        pe_matrix = torch.zeros(self.num_of_tokens, self.d_model)

        # 2. Fill the even columns (0, 2, 4...) with sine_matrix
        pe_matrix[:, 0::2] = sine_matrix

        # 3. Fill the odd columns (1, 3, 5...) with cosine_matrix
        pe_matrix[:, 1::2] = cosine_matrix
        
        output_tensor = input_tensor + pe_matrix

        return output_tensor
    
class MultiHeadAttention(nn.Module):
    """
    Compute Scaled Dot Product Attention using Multiple Heads
    """

    def __init__(self, d_model:int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Checking if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads # Dimension of query, key and value vectors within each head

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # Checking if input has correct embedding dimension
        if input.shape[-1] != self.d_model:
            raise ValueError(f"Expected input feature dimension to be d_model ({self.d_model}), but got {input.shape[-1]}")

        # Input Dimension: (batch_size, num_tokens, d_model)
        batch_size = input.shape[0]
        num_tokens = input.shape[1]

        # Creating query, key and value vectors from input embeddings. 
        # These vectors contain queries from all the heads combined.
        q = self.w_q(input)
        k = self.w_k(input)
        v = self.w_v(input)
        # Dimension: (batch_sizee, num_tokens, d_model)

        # Splitting our vectors into heads by reshaping our tensors.
        q_head = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k_head = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v_head = v.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        # Dimension: (batch_size, num_tokens, num_heads, head_dimension)

        # Swapping (num_tokens) and (num_heads) dimensions.
        # We do this because we need (num_tokens) and (head_dimension) at the end.
        # This is because we need to perform matrix operations on the last two dimensions
        q_head = q_head.transpose(1, 2)
        k_head = k_head.transpose(1, 2)
        v_head = v_head.transpose(1, 2)
        # Dimension: (batch_size, num_heads, num_tokens, head_dimension)

        # Getting a transpose of key vectors for matrix multiplication 
        k_t = k_head.transpose(-2, -1)
        sim1 = (q_head @ k_t)/math.sqrt(self.head_dim) # Scaled dot product of query and key vectors
        # sim1 is the matrix of attention scores for each token
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Performing softmax along the keys dimension to get attention scores for each key
        sim2 = F.softmax(sim1, dim=-1)
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Weighted sum of attention weights and value vectors
        sim3 = sim2 @ v_head
        # Dimension: (batch_size, num_heads, num_queries, head_dimension)

        # Swapping (num_heads) and (num_queries) dimension back so that we can recombine all the heads
        sim3 = sim3.transpose(1, 2).contiguous()
        sim3 = sim3.view(batch_size, num_tokens, self.d_model)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        # Passing all of our heads into a final layer so that all heads can interact.
        output = self.w_o(sim3)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        return output
    
class MaskedMultiHeadAttention(nn.Module):
    """
    Compute Masked Dot Product Attention using Multiple Heads
    """

    def __init__(self, d_model:int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Checking if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads # Dimension of query, key and value vectors within each head

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Checking if input has correct embedding dimension
        if input.shape[-1] != self.d_model:
            raise ValueError(f"Expected input feature dimension to be d_model ({self.d_model}), but got {input.shape[-1]}")

        # Input Dimension: (batch_size, num_tokens, d_model)
        batch_size = input.shape[0]
        num_tokens = input.shape[1]

        # Creating query, key and value vectors from input embeddings. 
        # These vectors contain queries from all the heads combined.
        q = self.w_q(input)
        k = self.w_k(input)
        v = self.w_v(input)
        # Dimension: (batch_sizee, num_tokens, d_model)

        # Splitting our vectors into heads by reshaping our tensors.
        q_head = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k_head = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v_head = v.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        # Dimension: (batch_size, num_tokens, num_heads, head_dimension)

        # Swapping (num_tokens) and (num_heads) dimensions.
        # We do this because we need (num_tokens) and (head_dimension) at the end.
        # This is because we need to perform matrix operations on the last two dimensions
        q_head = q_head.transpose(1, 2)
        k_head = k_head.transpose(1, 2)
        v_head = v_head.transpose(1, 2)
        # Dimension: (batch_size, num_heads, num_tokens, head_dimension)

        # Getting a transpose of key vectors for matrix multiplication 
        k_t = k_head.transpose(-2, -1)
        sim1 = (q_head @ k_t)/math.sqrt(self.head_dim) # Scaled dot product of query and key vectors
        # sim1 is the matrix of attention scores for each token
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Creating a matrix where lower triangle is 1, rest is 0
        mask = torch.tril(torch.ones(num_tokens, num_tokens))
        # Dimension: (num_queries, num_keys)

        # Applying the mask to our attention scores
        sim1 = sim1.masked_fill(mask == 0, float('-inf'))

        # Performing softmax along the keys dimension to get attention scores for each key
        sim2 = F.softmax(sim1, dim=-1)
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Weighted sum of attention weights and value vectors
        sim3 = sim2 @ v_head
        # Dimension: (batch_size, num_heads, num_queries, head_dimension)

        # Swapping (num_heads) and (num_queries) dimension back so that we can recombine all the heads
        sim3 = sim3.transpose(1, 2).contiguous()
        sim3 = sim3.view(batch_size, num_tokens, self.d_model)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        # Passing all of our heads into a final layer so that all heads can interact.
        output = self.w_o(sim3)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        return output
    
class CrossMultiHeadAttention(nn.Module):
    """
    Compute Scaled Dot Product Attention using Multiple Heads
    """

    def __init__(self, d_model:int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Checking if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads # Dimension of query, key and value vectors within each head

    def forward(self, encoder_output: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        
        # Checking if input has correct embedding dimension
        if encoder_output.shape[-1] != self.d_model:
            raise ValueError(f"Expected encoder input feature dimension to be d_model ({self.d_model}), but got {encoder_output.shape[-1]}")

        if decoder_input.shape[-1] != self.d_model:
            raise ValueError(f"Expected decoder input feature dimension to be d_model ({self.d_model}), but got {decoder_input.shape[-1]}")

        # Input Dimension: (batch_size, num_tokens, d_model)
        batch_size = decoder_input.shape[0]
        encoder_num_tokens = encoder_output.shape[1]
        decoder_num_tokens = decoder_input.shape[1]

        # Creating query, key and value vectors from input embeddings. 
        # These vectors contain queries from all the heads combined.
        q = self.w_q(decoder_input)
        k = self.w_k(encoder_output)
        v = self.w_v(encoder_output)
        # Dimension: (batch_sizee, num_tokens, d_model)

        # Splitting our vectors into heads by reshaping our tensors.
        q_head = q.view(batch_size, decoder_num_tokens, self.num_heads, self.head_dim)
        k_head = k.view(batch_size, encoder_num_tokens, self.num_heads, self.head_dim)
        v_head = v.view(batch_size, encoder_num_tokens, self.num_heads, self.head_dim)
        # Dimension: (batch_size, num_tokens, num_heads, head_dimension)

        # Swapping (num_tokens) and (num_heads) dimensions.
        # We do this because we need (num_tokens) and (head_dimension) at the end.
        # This is because we need to perform matrix operations on the last two dimensions
        q_head = q_head.transpose(1, 2)
        k_head = k_head.transpose(1, 2)
        v_head = v_head.transpose(1, 2)
        # Dimension: (batch_size, num_heads, num_tokens, head_dimension)

        # Getting a transpose of key vectors for matrix multiplication 
        k_t = k_head.transpose(-2, -1)
        sim1 = (q_head @ k_t)/math.sqrt(self.head_dim) # Scaled dot product of query and key vectors
        # sim1 is the matrix of attention scores for each token
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Performing softmax along the keys dimension to get attention scores for each key
        sim2 = F.softmax(sim1, dim=-1)
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Weighted sum of attention weights and value vectors
        sim3 = sim2 @ v_head
        # Dimension: (batch_size, num_heads, num_queries, head_dimension)

        # Swapping (num_heads) and (num_queries) dimension back so that we can recombine all the heads
        sim3 = sim3.transpose(1, 2).contiguous()

        # Recombining all the heads
        sim3 = sim3.view(batch_size, decoder_num_tokens, self.d_model)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        # Passing all of our heads into a final layer so that all heads can interact.
        output = self.w_o(sim3)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        return output

class FFNN(nn.Module):
    """
    Apply position-wise feed-forward network.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input feature dimension to be d_model ({self.d_model}), but got {x.shape[-1]}")

        z1 = self.hidden_layer(x)
        a1 = self.activation(z1)
        a1 = self.dropout(a1) # Randomly zeroes out some neurons to prevent overfitting
        output = self.output_layer(a1)

        return output
    
class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim 
        self.attention_layer = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffnn = FFNN(self.d_model, self.hidden_dim, dropout_rate=dropout_rate)
        self.norm1= nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Ouput of Positional Encoding
        # Dimension: (batch_size, num_tokens, d_model)
    
        # ----ATTENTION----
        # Applying Pre-Norm: Normalizing Before Residual Connection
        attn_input = self.norm1(x)
        # Normalizing First

        attn_output = self.attention_layer(attn_input)
        # Passing though attention layer
        
        x = x + self.dropout(attn_output)
        # Applying Residual Connection

        # ----FFNN----
        ffnn_input = self.norm2(x)
        # Normalizing
        ffnn_output = self.ffnn(ffnn_input)
        # Passing through network
        output = x + self.dropout(ffnn_output)
        # Applying Residual Connection
        
        # Dimension remains same: (batch_size, num_tokens, d_model)
        return output
    
class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim 
        self.cross_attention_layer = CrossMultiHeadAttention(self.d_model, self.num_heads)
        self.masked_attention_layer = MaskedMultiHeadAttention(self.d_model, self.num_heads)
        self.ffnn = FFNN(self.d_model, self.hidden_dim, dropout_rate=dropout_rate)
        self.norm1= nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # x: Ouput of Positional Encoding
        # Dimension: (batch_size, num_tokens, d_model)
    
        # ----ATTENTION----
        # Applying Pre-Norm: Normalizing Before Residual Connection
        attn_input = self.norm1(x)
        # Normalizing First

        attn_output = self.masked_attention_layer(attn_input)
        # Passing though attention layer
        
        x = x + self.dropout(attn_output)
        # Applying Residual Connection

        # ----CROSS ATTENTION----
        cross_attn_input = self.norm2(x)
        # Normalizing

        cross_attn_output = self.cross_attention_layer(encoder_output, cross_attn_input)
        # Passing though cross attention layer

        x = x + self.dropout(cross_attn_output)
        # Applying Residual Connection

        # ----FFNN----
        ffnn_input = self.norm3(x)
        # Normalizing
        ffnn_output = self.ffnn(ffnn_input)
        # Passing through network
        output = x + self.dropout(ffnn_output)
        # Applying Residual Connection
        
        # Dimension remains same: (batch_size, num_tokens, d_model)
        return output


# Now, Stack Multiple Encoders:
class EncoderStack(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, num_blocks: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_blocks)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dimension of tokens: (batch_size, num_tokens) 

        for layer in self.layers:
            x = layer(x)
        output = self.norm_1(x)
        # Dimension: (Batch_size, num_tokens, d_model)

        return output
    
# Now, Stack Multiple Decoders:
class DecoderStack(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, num_blocks: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_blocks)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # Dimension of tokens: (batch_size, num_tokens)

        for layer in self.layers:
            x = layer(x, encoder_output)
        output = self.norm_1(x)
        # Dimension: (Batch_size, num_tokens, d_model)

        return output
    

# Cross-Attention: A Dimensional Analysis Guide

This guide breaks down the Cross-Attention mechanism of the Transformer architecture through rigorous dimensional analysis. By tracking the exact shape of the tensors at every step, the abstract linear algebra translates into a logical flow of information between the Decoder and the Encoder.

---

## 1. The Starting Variables

To make the math concrete, this walkthrough uses the following baseline variables, noting the distinct sequence lengths for the Encoder and Decoder:

- **`batch_size` = 2** (2 independent sequences processed simultaneously)
    
- **`decoder_num_tokens` = 3** (3 tokens currently in the Decoder sequence)
    
- **`encoder_num_tokens` = 5** (5 tokens in the original Encoder sequence)
    
- **`d_model` = 8** (The total embedding size: 8 features per word)
    
- **`num_heads` = 2** (The number of parallel attention heads/specialists)
    
- **`head_dim` = 4** (Calculated as $d_{\text{model}} \div \text{num\_heads} = 4$ features per head)
    

---

## 2. The Complete Dimensional Flow (Step-by-Step)

## Step 1: The Inputs

Cross-attention requires two separate streams of data.

- **Decoder Input Shape:** `(batch_size, decoder_num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Encoder Input Shape:** `(batch_size, encoder_num_tokens, original_features)` $\rightarrow$ **`(2, 5, 8)`**
    
- **Meaning:** We have a batch of 2 sequences. The Decoder is processing 3 tokens, while the Encoder's memory contains 5 tokens. Every token has an 8-feature dossier.
    

## Step 2: The Mega-Matrix Projections ($W^Q$, $W^K$, $W^V$)

The Queries are generated from the Decoder, while the Keys and Values are pulled from the Encoder.

- **$Q$ Shape (Decoder):** `(batch_size, decoder_num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **$K$, $V$ Shape (Encoder):** `(batch_size, encoder_num_tokens, original_features)` $\rightarrow$ **`(2, 5, 8)`**
    
- **Meaning:** The Decoder's 3 tokens ask questions (Queries), while the Encoder's 5 tokens provide the context (Keys and Values).
    

## Step 3: Splitting into Heads (`.view()`)

We distribute the data to our multiple "specialists" (attention heads).

- **$Q$ Shape:** `(batch_size, decoder_num_tokens, num_heads, head_dim)` $\rightarrow$ **`(2, 3, 2, 4)`**
    
- **$K$, $V$ Shape:** `(batch_size, encoder_num_tokens, num_heads, head_dim)` $\rightarrow$ **`(2, 5, 2, 4)`**
    
- **Meaning:** We split the 8 features into 2 distinct heads, each holding 4 isolated features.
    

## Step 4: Aligning for Parallel Math (`.transpose(1, 2)`)

Matrix multiplication (`@` in PyTorch) requires swapping the tokens and heads dimensions.

- **$Q$ Shape:** `(batch_size, num_heads, decoder_num_tokens, head_dim)` $\rightarrow$ **`(2, 2, 3, 4)`**
    
- **$K$, $V$ Shape:** `(batch_size, num_heads, encoder_num_tokens, head_dim)` $\rightarrow$ **`(2, 2, 5, 4)`**
    
- **Meaning:** We now have 2 independent heads. PyTorch pushes `num_heads` out of the last two slots so the math computes in parallel.
    

## Step 5: The Cross-Attention Dot Product ($Q \cdot K^T$)

We multiply the Decoder's Queries by the Encoder's Keys to get raw compatibility scores.

- **$Q$ Shape:** **`(2, 2, 3, 4)`**
    
- **$K^T$ Shape:** (`k.transpose(-2, -1)`) $\rightarrow$ **`(2, 2, 4, 5)`**
    
- **Matrix Math:** $3 \times 4$ multiplied by $4 \times 5$ results in $3 \times 5$.
    
- **`scores` Shape:** `(batch_size, num_heads, num_queries, num_keys)` $\rightarrow$ **`(2, 2, 3, 5)`**
    
- **Meaning:** Inside both heads, we have a $3 \times 5$ grid of raw scores. Every single one of the **3** Decoder tokens has calculated its compatibility with all **5** of the Encoder tokens.
    

## Step 6: The Softmax

We apply `F.softmax(scores, dim=-1)` to target the last dimension (`num_keys`).

- **`weights` Shape:** **`(2, 2, 3, 5)`**
    
- **Meaning:** The shape remains unchanged. For every individual head, each of the 3 Decoder Queries has now distributed 100% of its attention budget across the 5 Encoder Keys.
    

## Step 7: Applying Attention to Values ($Weights \cdot V$)

The Decoder Queries use their calculated percentages to grab pieces of the Encoder's Value vectors.

- **`weights` Shape:** **`(2, 2, 3, 5)`**
    
- **$V$ Shape:** **`(2, 2, 5, 4)`**
    
- **Matrix Math:** $3 \times 5$ multiplied by $5 \times 4$ results in $3 \times 4$.
    
- **`context` Shape:** `(batch_size, num_heads, num_queries, head_dim)` $\rightarrow$ **`(2, 2, 3, 4)`**
    
- **Meaning:** The 5-token Encoder dimension is completely consumed. Our 3 Decoder tokens now hold 4 features of perfectly blended context derived from the Encoder sequence.
    

## Step 8: Taping the Dossier Back Together

We reverse the alignment and glue the independent heads back into a single continuous vector.

- **`context.transpose(1, 2)`:** Swaps `num_heads` and `num_queries` $\rightarrow$ **`(2, 3, 2, 4)`**
    
- **`.contiguous().view(...)`:** Crushes the last two dimensions together ($2 \times 4 = 8$).
    
- **Flattened Shape:** `(batch_size, decoder_num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** The 2 specialists have returned their 4 pages each. Our 3 Decoder tokens are back to having 8 features, now deeply enriched by the Encoder's perspective.
    

## Step 9: The Output Projection ($W^O$)

Finally, the flattened tensor passes through the final linear layer.

- **Output Shape:** `(batch_size, decoder_num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** The tensor returns to the exact shape of the Decoder input, ready for the next block.
    

---

## 3. Common Misunderstandings & Clarifications

## Clarification 1: The "Double Transpose" of the Keys Matrix ($K$)

When looking at the code, it can be confusing to see the $K$ matrix seemingly transposed twice. These are two separate operations serving completely different mathematical rules:

1. **The "Parallelization" Transpose (`.transpose(1, 2)`):**
    
    - Applied to $Q$, $K$, and $V$.
        
    - Changes shape from `(batch, tokens, heads, head_dim)` to `(batch, heads, tokens, head_dim)`.
        
    - **Why:** PyTorch's `@` operator uses the last two dimensions for matrix math. If we don't move `heads` out of the way, PyTorch will try to multiply tokens by heads, which is mathematically invalid.
        
2. **The "Matrix Math" Transpose (`.transpose(-2, -1)`):**
    
    - Applied **ONLY** to $K$ during the dot product ($Q \cdot K^T$).
        
    - Changes the last two dimensions of $K$ from $5 \times 4$ to $4 \times 5$.
        
    - **Why:** You cannot multiply a $3 \times 4$ matrix by a $5 \times 4$ matrix. The inner dimensions must match for the dot product to execute.
        

## Clarification 2: The Purpose of the Final Output Layer ($W^O$)

When we use `.view()` to flatten the heads in Step 8, we are just placing the 4 features from Head 1 next to the 4 features from Head 2 in an array: `[H1, H1, H1, H1, H2, H2, H2, H2]`.

At this point, the data from the different heads hasn't actually interacted. The $W^O$ layer (a trainable $8 \times 8$ weight matrix) acts as the **mixer**. By multiplying the concatenated array by this weight matrix, every single one of the 8 output numbers becomes a mathematically blended combination of _both_ heads. It synthesizes the isolated insights into a single, cohesive representation of the token.

## Clarification 3: Why There is No Causal Mask

Unlike Masked Self-Attention, Cross-Attention does not use a causal (lower-triangular) mask. Causal masking is used to prevent the model from looking at future tokens it hasn't generated yet. In Cross-Attention, the Decoder is looking at the Encoder's output. Since the entire original input sequence (e.g., a sentence being translated) is already fully known and processed by the Encoder, the Decoder is completely allowed to look at every token in that sequence simultaneously to gather the best context.
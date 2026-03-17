# Multi-Head Attention (MHA): A Dimensional Analysis Guide

This guide breaks down the Multi-Head Attention mechanism of the Transformer architecture through rigorous dimensional analysis. By tracking the exact shape of the tensors at every step, the abstract linear algebra translates into a logical flow of information.

---

## 1. The Starting Variables

To make the math concrete, this walkthrough uses the following baseline variables:

- **`batch_size` = 2** (2 independent sequences processed simultaneously)
    
- **`num_tokens` = 3** (3 words/tokens per sequence)
    
- **`d_model` = 8** (The total embedding size: 8 features per word)
    
- **`num_heads` = 2** (The number of parallel attention heads/specialists)
    
- **`head_dim` = 4** (Calculated as $d_{\text{model}} \div \text{num\_heads} = 4$ features per head)
    

---

## 2. The Complete Dimensional Flow (Step-by-Step)

## Step 1: The Input

We start with the raw, embedded sequence data.

- **Shape:** `(batch_size, num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** A batch of 2 sequences. Each sequence has 3 tokens. Each token is described by an 8-feature vector (the "dossier").
    

## Step 2: The Mega-Matrix Projections ($W^Q$, $W^K$, $W^V$)

Instead of projecting down to a smaller size right away, the input passes through `nn.Linear(d_model, d_model)` layers to create the Queries, Keys, and Values.

- **$Q$, $K$, $V$ Shape:** `(batch_size, num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** Every word now has an 8-feature Query, an 8-feature Key, and an 8-feature Value. At this stage, the data for all attention heads is glued together in a single vector.
    

## Step 3: Splitting into Heads (`.view()`)

We distribute the data to our multiple "specialists" (attention heads) by slicing the `original_features` dimension into `num_heads` and `head_dim`.

- **$Q$, $K$, $V$ Shape:** `(batch_size, num_tokens, num_heads, head_dim)` $\rightarrow$ **`(2, 3, 2, 4)`**
    
- **Meaning:** For our 3 tokens, we now have 2 distinct heads, each holding 4 isolated features.
    

## Step 4: Aligning for Parallel Math (`.transpose(1, 2)`)

Matrix multiplication (`@` in PyTorch) only operates on the last two dimensions. We must swap the `num_tokens` and `num_heads` dimensions.

- **$Q$, $K$, $V$ Shape:** `(batch_size, num_heads, num_tokens, head_dim)` $\rightarrow$ **`(2, 2, 3, 4)`**
    
- **Meaning:** We now have 2 independent heads. Inside each head, 3 tokens are holding 4 features each. PyTorch pushes `num_heads` out of the last two slots, treating it like an extra batch dimension so the math computes in parallel.
    

## Step 5: The Dot Product ($Q \cdot K^T$)

We multiply Queries by Keys to get raw compatibility scores. To align the inner dimensions for matrix multiplication, we transpose $K$'s last two dimensions.

- **$Q$ Shape:** **`(2, 2, 3, 4)`**
    
- **$K^T$ Shape:** (`k.transpose(-2, -1)`) $\rightarrow$ **`(2, 2, 4, 3)`**
    
- **Matrix Math:** $3 \times 4$ multiplied by $4 \times 3$ results in $3 \times 3$.
    
- **`scores` Shape:** `(batch_size, num_heads, num_queries, num_keys)` $\rightarrow$ **`(2, 2, 3, 3)`**
    
- **Meaning:** Inside both heads independently, we have a $3 \times 3$ grid of raw compatibility scores between the Queries and the Keys.
    

## Step 6: The Softmax

We apply `F.softmax(scores, dim=-1)` to target the last dimension (`num_keys`).

- **`weights` Shape:** **`(2, 2, 3, 3)`**
    
- **Meaning:** The shape remains unchanged. For every individual head, every Query has now distributed 100% of its attention budget across the 3 Keys.
    

## Step 7: Applying Attention to Values ($Weights \cdot V$)

The Queries use their calculated percentages to grab pieces of the Value vectors.

- **`weights` Shape:** **`(2, 2, 3, 3)`**
    
- **$V$ Shape:** **`(2, 2, 3, 4)`**
    
- **Matrix Math:** $3 \times 3$ multiplied by $3 \times 4$ results in $3 \times 4$.
    
- **`context` Shape:** `(batch_size, num_heads, num_queries, head_dim)` $\rightarrow$ **`(2, 2, 3, 4)`**
    
- **Meaning:** The Keys dimension is consumed. In both heads, our 3 Queries now hold 4 features of perfectly blended context.
    

## Step 8: Taping the Dossier Back Together

We reverse the alignment from Step 4 and the split from Step 3 to glue the independent heads back into a single continuous vector.

- **`context.transpose(1, 2)`:** Swaps `num_heads` and `num_queries` $\rightarrow$ **`(2, 3, 2, 4)`**
    
- **`.contiguous().view(...)`:** Crushes the last two dimensions together ($2 \times 4 = 8$).
    
- **Flattened Shape:** `(batch_size, num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** The 2 specialists have returned their 4 pages each, which are taped side-by-side. Our 3 tokens are back to having 8 features, enriched by multiple perspectives.
    

## Step 9: The Output Projection ($W^O$)

Finally, the flattened tensor passes through the final linear layer.

- **Output Shape:** `(batch_size, num_tokens, original_features)` $\rightarrow$ **`(2, 3, 8)`**
    
- **Meaning:** The tensor returns to its exact original shape, ready to be passed into the Feed-Forward Network or the next Transformer block.
    

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
        
    - Changes the last two dimensions of $K$ from $3 \times 4$ to $4 \times 3$.
        
    - **Why:** You cannot multiply a $3 \times 4$ matrix by a $3 \times 4$ matrix. The inner dimensions must match for the dot product to execute.
        

## Clarification 2: The Purpose of the Final Output Layer ($W^O$)

If the flattened tensor goes into $W^O$ as shape `(2, 3, 8)` and comes out as `(2, 3, 8)`, why run it through a neural network layer at all?

When we use `.view()` to flatten the heads in Step 8, we are just placing the 4 features from Head 1 next to the 4 features from Head 2 in an array: `[H1, H1, H1, H1, H2, H2, H2, H2]`.

At this point, the data from the different heads hasn't actually interacted.

The $W^O$ layer (a trainable $8 \times 8$ weight matrix) acts as the **mixer**. By multiplying the concatenated array by this weight matrix, every single one of the 8 output numbers becomes a mathematically blended combination of _both_ heads. It synthesizes the isolated insights into a single, cohesive representation of the token.
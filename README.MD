# GEOGPTnano - A course GPT 
This GPT is a inspired by https://github.com/karpathy/nanoGPT.git and https://github.com/jbxamora/reversenanogpt.git

## Quick Start
1.Clone the repo

2.Visit https://wortschatz-leipzig.de/en/download/kat#kat-ge_web_2019 and download following:

kat_news_2020_100K-sentences.txt

kat_newscrawl_2016_1M-sentences.txt

kat-ge_web_2019_1M-sentences.txt

Put them under data/raw

3.run python -m src.train --config configs/train_small.yaml --resume best_step8000_val3.862.pt to train.

## Full Train
1.Clone the repo

2.Visit https://wortschatz-leipzig.de/en/download/kat#kat-ge_web_2019 and download following:

kat_news_2020_100K-sentences.txt

kat_newscrawl_2016_1M-sentences.txt

kat-ge_web_2019_1M-sentences.txt

Put them under data/raw

3.run python.exe src/00_data_process.py --config configs/data.yaml to clean and split txt.

4.run python.exe src/01_tokenizer.py --config configs/tokenizer.yaml to generate tokenizer.

5.run python.exe src/train.py --config configs/train_smaill.yaml to train.

## Dataset

* **Source:** Leipzig Corpora Collection — Georgian parts
* **Size (raw):** ~582 MB
* **Format:** line-based `*.txt` aggregated post-cleaning
* **Analysis:** please see the notebooks\data_analysis.ipynb

## Cleaning Pipeline

1. **Drop empties** (after trim).
2. **Strip leading line numbers** (common in source)(at first forget to clean them, then my GPT is always obssesed with random numbers):
   `^\s*\d+(?:[.)]|:)?\s*(?:[!\"“”„\-–—])?\s*` → remove once at line start.
3. **Normalize & tidy:** Unicode **NFKC** (optional lowercase), remove **HTML** tags and **URLs**, collapse whitespace.
4. **Quality filters (all must pass):**

   * `min_chars ≥ 15` (counting spaces)
   * `digits_ratio ≤ 0.40` = digits / chars (no spaces)
   * `georgian_ratio ≥ 0.85` = Georgian letters / alphabetic letters
     (Georgian ranges: U+10A0–U+10FF, U+1C90–U+1CBF)
5. **De-dup:**

   * **Exact:** MD5 on fully cleaned line (cross-file, first wins)
   * **Near-dup (optional):** bucket by first 16 chars; 5-gram Jaccard ≥ 0.90 → drop
6. **Post:** append EOS token (default `<eos>`).
7. **Split:** deterministic train/val/test = **98/1/1** (seed 1337; optional shuffle).

## Outputs

* **Merged cleaned corpus** (one line per sample)
* **Splits:** `train.txt`, `val.txt`, `test.txt`
* **Stats (JSON):** per-file counts, drop reasons, exact/near-dup tallies, split sizes


## Tokenizer
In the notebooks\data_analysis.ipynb I tried 3 tokenizers with different sizes. Though the 32k ver. may contain more words, it may cost too much and not worthy for our model.

[16000] -> {'lines': 20609, 'avg_subwords_per_sent': 25.894609151341648, 'p50': 23.0, 'p90': 42.0, 'p99': 64.0, 'tokens_per_char': 0.228380709549835}

[24000] -> {'lines': 20609, 'avg_subwords_per_sent': 24.459265369498763, 'p50': 22.0, 'p90': 39.0, 'p99': 60.0, 'tokens_per_char': 0.2157215174597224}

[32000] -> {'lines': 20609, 'avg_subwords_per_sent': 23.61895288466204, 'p50': 21.0, 'p90': 38.0, 'p99': 58.0, 'tokens_per_char': 0.20831027752136433}

## Model

* **Type:** decoder-only Transformer (RMSNorm, RoPE, GELU, weight tying)
* **Compat:** checkpoint-shape stable if `vocab_size, n_layer, n_head, d_model, ffn_mult` unchanged
* **Symbols:** `B` batch, `T` seq len, `C` model dim, `H` heads, `hd=C/H` (must be even)

## Forward Pipeline 

1. **Embed:** token IDs → `B×T×C`, dropout
2. **Blocks × N:**

   * RMSNorm → **Self-Attention**

     * Linear QKV (`3C`) → reshape to `(B,H,T,hd)`
     * **RoPE** on `q,k` (cached cos/sin; auto-extend)
     * **SDPA** (`is_causal=True`) on CUDA if enabled; else masked softmax with causal mask
     * Projection `C→C`, dropout, **residual**
   * RMSNorm → **MLP** (`C→ffn_mult·C→GELU→C`), dropout, **residual**
3. **Final RMSNorm**
4. **LM head:** `C→V` (tied to embedding weights if enabled)

## Training

* **Loss:** cross-entropy over `logits.view(-1,V)` vs `labels.view(-1)`, `ignore_index=-100`
* **Label shift:** external; model does not shift targets internally
* **Dropout:** on attn probs, projections, MLP outputs (train only)

## Generation

* Condition on last `n_ctx` tokens; iterate: forward → temperature scale → **top-p** (optional) or multinomial → append
* **No KV cache** (per-step O(T²)); candidate for optimization

## Positional Encoding

* **RoPE** (`theta=10000`) with `(cos,sin)` buffers; buffers are non-persistent (not saved in checkpoints)

## Toggles

* `use_sdpa`: use PyTorch SDPA/Flash path on CUDA (`is_causal=True`)
* `grad_checkpoint`: per-block gradient checkpointing
* `tie_weights`: share `tok_emb` ↔ `lm_head`
* `bias`: enable Linear biases (default off)

## Constraints & Edge Cases

* **Non-SDPA path:** requires `T ≤ n_ctx` (fixed causal mask)
* **RoPE:** `hd` must be even
* **CPU:** always uses masked softmax path
* **Checkpointing:** RoPE caches are non-persistent; weight shapes unchanged across the new toggles

## Training Process(my bad to not have recorded the train log)
1.Started with **lr=1.5e-3**,The training process was smooth before the **step 2500**, then it came to an overfitting stage which I then stopped to avoid.
2.Continued from the checkpoint-2500, this time with dropout_train=0.10, and the learning rate reduced to half **lr=7.5e-4**. This time the val_loss reduced to around 3.90 , maintained a ~0.4 difference with train_loss.
3.Then from step 6050 enabled the swa with swa_freq=500 and swa_lr = 5e-4 to try to let the model converge better, and it did, val_loss reduced to **~3.86**
4.After that, reduced the swa_freq to make swa work more often, reduced lr=4e-4, yet this the eval_loss showed a slight increase and there's where I decided to stop.

## Model Evaluation
to see clearly, please see the eval.ipynb, I tried a little smoke test there.
The model works relatively poor still on the basic questions, yet the text consistency(around 60%) is acceptable for so-far training and such a "tiny" model with 50M parameters.
For SWA comparison, in the same notebook I tried with checkpoint-6000(best before SWA), and checkpoint-8000(best after SWA), it turns out that basically 8000 model won the consistency(by 0.2) and ask-answer section(often answers 2-3 ok, 6000 only 1).

## Enviroment
Trained on my 4070TIS, lasted about 4 hours from 0 to step 8000.

## Access
To access the model trained, please visit Aurelianous/GeoGPTnano on Huggingface to download the best checkpoint so far

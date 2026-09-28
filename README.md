# Masked Language Model Toolkit

An interactive command-line tool for probing pre-trained BERT-family masked language
models: predict masked tokens, inspect self-attention head by head, project contextual
embeddings into 2D, attribute predictions with SHAP, and fine-tune on your own text.

Five languages are supported, each backed by its own monolingual checkpoint:

| Language | Checkpoint |
|----------|------------|
| English  | `bert-base-uncased` |
| French   | `camembert-base` |
| German   | `dbmdz/bert-base-german-uncased` |
| Chinese  | `bert-base-chinese` |
| Japanese | `cl-tohoku/bert-base-japanese` |

The tool loads one checkpoint per session, chosen at startup. It is a multi-language
tool rather than a single multilingual model — nothing is shared across languages.

## Install

```bash
git clone https://github.com/knb2345/Multilingual-Masked-Language-Model.git
cd Multilingual-Masked-Language-Model
pip install -r requirements.txt
```

TensorFlow (via `TFAutoModelForMaskedLM`) does the modelling; `tf-keras` is required
because Transformers does not yet support Keras 3. The first run of each language
downloads its checkpoint from the Hugging Face Hub.

## Run

```bash
python mask.py
```

You are prompted for a language, then given a menu:

### 1. Predict masked token

Enter a sentence containing exactly one `[MASK]` token. The tool prints the top 3
candidates with softmax confidences and writes an attention heatmap for **every**
layer/head pair to `images/Attention_Layer{L}_Head{H}.png` — token labels on both axes,
cell brightness proportional to attention weight. For `bert-base-uncased` that is 12
layers x 12 heads = 144 images, which is the point: individual heads specialise, and
seeing them side by side is how you find the ones that track syntax or coreference.

```
Enter text with [MASK] token: The capital of France is [MASK].
Prediction: paris (Confidence: 0.4213)
...
```

Create the `images/` directory before running this option.

### 2. Fine-tune model

Point the tool at a plain-text file and give it an epoch count. The text is tokenised,
chunked into fixed-length blocks, and trained with `DataCollatorForLanguageModeling`
(dynamic random masking) on a 90/10 train/validation split. Perplexity is measured on a
held-out probe set before and after training, so adaptation is reported as a number
rather than asserted.

### 3. Analyze contextual embeddings

Takes a sentence, pulls final-layer hidden states, and projects them to 2D — PCA under
50 tokens, t-SNE above (perplexity capped at `n-1`). Saved to
`contextual_embeddings.png`, coloured by token position. Useful for showing that the
same surface word gets different vectors in different contexts.

### 4. Explain prediction

Runs a SHAP `Explainer` over a text masker with the model's own top-3 candidates as
output names, so you get per-input-token attributions for each candidate: which words in
the sentence actually pushed the model toward `paris` rather than `lyon`. This is the
slowest option — SHAP re-runs the model over many perturbations of the input.

## Repository layout

```
mask.py            # everything: EnhancedMaskedLanguageModel class + CLI
requirements.txt
assets/fonts/      # OpenSans, used to label attention diagrams
```

## Known limitations

- One `[MASK]` per sentence; multi-mask inputs are not handled.
- No quantitative accuracy evaluation of predictions — the tool is for inspection, not
  benchmarking.
- Attention visualisation writes every head to disk with no filtering; expect a few
  hundred PNGs per sentence.
- Fine-tuning holds the whole corpus in memory, so it suits small domain corpora.

## Acknowledgements

The attention-diagram rendering (PIL grid, per-head PNG output) follows the structure of
Harvard's CS50 AI `attention` project; the multi-language support, fine-tuning,
perplexity benchmarking, embedding projection, and SHAP explainability are built on top
of it.

# PhonoBench

Code for **PhonoBench: Evaluating Sign-Form Reasoning in Video MLLMs**.

PhonoBench represents signs using 17 ASL-LEX annotation fields, connects signs
that differ in exactly one field, and generates questions about those relations.
The code generates tasks from Sem-Lex metadata and videos obtained by each
researcher. **No Sem-Lex data are included in this repository.** Generated
questions, predictions, embeddings, and adapters are produced locally.

## Repository

| Path | Contents |
| --- | --- |
| `src/graph/` | Metadata loading, graph construction, ten formal task programs and symbolic answers |
| `src/prompts/` | Task templates, feature definitions and label descriptions |
| `src/qwen/` | Qwen inference, frozen embeddings and feature-adapter training |
| `src/internvl/` | InternVL inference with synchronized video panels |
| `src/video.py` | Video paths, decoding, sampling and visual controls |
| `src/parsing.py`, `src/metrics.py` | Answer parsing and evaluation |
| `resources/semlex_metadata.csv` | User-provided Sem-Lex metadata (not distributed) |
| `resources/tasks/` | Generated graphs and task files |
| `experiments/` | Task generation, model runs, analysis, feature evaluation and linear probing |
| `tests/` | Small checks using synthetic data |

The code uses ordinary functions, dictionaries, lists and explicit loops.
PyTorch handles model inference, adapter training and linear probes.

## Data access

**Download Sem-Lex directly from its maintainers.** This repository distributes
PhonoBench code, not Sem-Lex videos, Sem-Lex metadata, or the ASL-LEX database.

1. Visit the [official Sem-Lex repository](https://github.com/leekezar/SemLex).
   Follow its **terms-of-use** link, agree to the applicable conditions, and
   obtain the dataset and its associated video-to-lexical-sign metadata through
   the original distribution channel.
2. Copy your authorized local copy of the Sem-Lex metadata table to
   `resources/semlex_metadata.csv` (rename the file locally if necessary):

   ```bash
   mkdir -p resources
   cp /path/to/downloaded/semlex_metadata.csv resources/semlex_metadata.csv
   ```

3. Keep videos in a separate local directory and pass that directory with
   `--video-root` to the commands below. Video paths do not need to be stored
   inside this repository.

The metadata CSV must contain `video_id`, `signer_id`, `split`, `label_type`,
`label`, and the 17 ASL-LEX fields listed in `src/graph/fields.py`, with one
row for each video annotation. The video-to-lexical-sign alignments are part of
Sem-Lex; **the standalone ASL-LEX database is not a substitute** for this table.
If your downloaded release uses a different CSV schema, map its columns to the
names above locally before running the code. Do not commit your downloaded
metadata or video files to this repository.

Task counts are determined by the metadata and available video files on each
machine; the code does not require a particular number of videos or graphs.

## Installation

Use Python 3.11 or 3.12. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a compatible PyTorch and torchvision pair using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Graph construction, task generation and result analysis use only the Python
standard library. Model inference needs the model weights and suitable hardware.
The default adapter-training command uses CUDA, 4-bit NF4 weights and bfloat16
computation. Use `--no-quantization` to train without 4-bit loading.

## 1. Prepare the videos

After obtaining the data as described above, pass the directory containing
your local Sem-Lex videos as `--video-root`.
The resolver accepts these layouts, where `<id>` is a `video_id` from the CSV:

```text
<video-root>/<split>/<id>.mp4
<video-root>/<split>/<id>/videos/rgb_clean_480p.mp4
<video-root>/<id>.mp4
<video-root>/<id>/videos/rgb_clean_480p.mp4
```

It also accepts `rgb_480.mp4` and the same filenames directly inside the
video's directory. Use a root containing both `train` and `test` for feature
training and probing. A root containing only test videos is sufficient for
graph tasks.

The graph uses ASL-LEX-labelled test examples. It checks which videos exist,
groups their annotations by lexical sign, and retains complete, consistent
17-field vectors. Ambiguous video identities and conflicting vectors are
excluded. The category `None` is retained as an annotation value. Feature
prediction and training need only the requested field, so unrelated missing
annotations do not discard an example.

## 2. Generate tasks

```bash
python -m experiments.build \
  --video-root /path/to/semlex \
  --experiment benchmark \
  --count 1000 \
  --seed 17
```

This writes a graph, formal programs, `benchmark.jsonl` and a generation summary
under `resources/tasks/generated/`. Each JSONL row contains a question, ordered
video roles, answer choices and a symbolically computed gold answer.

`--count` is an upper bound per formal task. Available videos, annotations and
graph motifs determine how many examples can be generated. The generator does
not duplicate examples to meet the requested limit. Use `--output` to choose
another directory, or `--overwrite` to regenerate the known task files there.

The ten formal tasks are:

| Task | Operation |
| --- | --- |
| `same_sign` | Decide whether two videos represent the same lexical sign |
| `one_feature_pair` | Decide whether two distinct signs differ in exactly one field |
| `distinguishing_feature` | Name the unique differing field |
| `exists_one_feature_pair` | Find whether a set contains a one-field pair |
| `universal_shared_feature` | Decide whether all signs share a named field value |
| `feature_value_counting` | Count signs with a specified field value |
| `constraint_satisfaction` | Select a sign satisfying two named relations |
| `feature_transformation` | Apply a demonstrated field change to another sign |
| `common_neighbor` | Select a sign adjacent to two given signs |
| `isolate_detection` | Select the sign with no one-field neighbor in the set |

For symbolic work without video files, pass `--metadata-only`. Build again with
`--video-root` before evaluating models so the tasks use your available media.

## 3. Run a model and analyze predictions

Check the input paths without loading weights:

```bash
python -m experiments.run \
  --manifest resources/tasks/generated/benchmark.jsonl \
  --video-root /path/to/semlex \
  --dry-run
```

Run Qwen and evaluate its predictions:

```bash
python -m experiments.run \
  --manifest resources/tasks/generated/benchmark.jsonl \
  --video-root /path/to/semlex \
  --model qwen \
  --output experiments/results/qwen.jsonl

python -m experiments.analyze --results experiments/results/qwen.jsonl
```

Use `--model internvl` for InternVL. The supported checkpoint settings are:

| Runner | Checkpoints | Video input |
| --- | --- | --- |
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` (default), `Qwen/Qwen3-VL-32B-Instruct` | Separate video items, normally 32 uniformly sampled frames per source |
| `internvl` | `OpenGVLab/InternVL3-8B-hf` (default), `OpenGVLab/InternVL3-14B-hf` | One synchronized composite containing 28 frames per source, panels in prompt order |

Select a model with `--checkpoint` and pin its version with `--revision`.
Both runners use deterministic generation. Qwen uses a maximum of 307,200
pixels per sampled frame. The default generation limit is 32 new tokens.
`--dtype`, `--device`, `--attention` and `--max-new-tokens` are configurable.
The default attention implementation is `sdpa`; FlashAttention is optional.

Use `--task`, `--condition`, `--field` or `--limit` to select a subset.
Every run saves the selected questions in a sibling `.inputs.jsonl` file and
the model configuration in `.run.json`. Analysis uses those exact inputs as
its denominator. Missing, invalid and failed predictions count as incorrect.
It reports accuracy, response validity, chance-normalized accuracy and the
paired or per-field measures applicable to the selected experiment.

To continue an interrupted run, repeat its command with `--resume`. Add
`--retry-errors` to retry technical failures. Resume checks that the inputs,
settings and model environment match. `--bootstrap 1000` on the analysis
command adds confidence intervals using the task cluster IDs.

## 4. Controlled experiments

Generate one diagnostic family using `--experiment NAME`, or generate all
families together:

```bash
python -m experiments.build \
  --video-root /path/to/semlex \
  --experiment all \
  --count 1000 \
  --output resources/tasks/all

python -m experiments.run \
  --manifest resources/tasks/all/grounding.jsonl \
  --video-root /path/to/semlex \
  --output experiments/results/grounding.jsonl

python -m experiments.analyze --results experiments/results/grounding.jsonl
```

| Experiment | Questions and conditions |
| --- | --- |
| `matched_pairs` | Identity, one-field and named-field retrieval on the same candidate videos; candidate swaps |
| `grounding` | Atomic feature values, equality, sole-difference verification and field attribution |
| `counterfactuals` | Fixed questions with video substitutions that flip the correct answer |
| `visual_controls` | Original, black, absent, repeated-frame and deranged video inputs |
| `evidence` | Frame budget, signing-space crop, motion-based sampling, temporal shuffle and reversal |
| `binding` | Target positions, congruent/conflicting distractors, universal sets and stream masking |
| `token_robustness` | Different video productions and same/cross-signer pairs for the same lexical edge |
| `interface` | Field names and definitions, anonymous properties, answer rotations and video order |
| `gold_features` | Text tables containing full or task-specific gold feature values |
| `atomic_vectors` | All 17 field predictions per video, followed by relation reconstruction |

For diagnostic families, `--count` caps base examples before matched conditions
expand into multiple questions. For `gold_features`, the cap applies per formal
task. For `binding`, it applies separately to target-feature and universal-set
examples. The generation summary reports the resulting question counts.
Use Qwen3-VL-8B for the controlled experiments.
The signing-space crop combines face and motion detections into one padded box
fixed across the clip, with a full-frame fallback when neither is found. The
actual crop and sampled frame indices are saved with each prediction.

## 5. Feature prediction and adaptation

Evaluate a single annotation field:

```bash
python -m experiments.features \
  --field Contact \
  --video-root /path/to/semlex \
  --output experiments/results/features/contact/base
```

Train one adapter on the training split, then evaluate it on the test split:

```bash
python -m src.qwen.train \
  --field Contact \
  --video-root /path/to/semlex \
  --output experiments/results/adapters/contact

python -m experiments.features \
  --field Contact \
  --video-root /path/to/semlex \
  --adapter experiments/results/adapters/contact \
  --load-4bit \
  --output experiments/results/features/contact/adapter
```

Run one field at a time; names containing spaces must be quoted. The field
list is `FEATURE_FIELDS` in `src/graph/fields.py`. Allowed labels come from the
metadata's ASL-LEX annotation vocabulary. Training examples use only the train
split. Video and signer overlap between the train and test examples is rejected.

The training defaults are one epoch, learning rate `2e-4`, accumulation of
8 single-video microbatches, rank 8, alpha 16 and zero LoRA dropout. Only
assistant answer tokens contribute to the training loss. `--targets attention`
adapts language attention projections; `vision`, `projector` and
`vision_language` select the other adaptation experiments. The loop uses AdamW
with weight decay `0.01`. Use `--help` for the complete settings.

### Transfer to graph tasks

Evaluate the base model and the field adapter on the same generated questions.
`--field` retains programs that explicitly use the requested field.

```bash
python -m experiments.run \
  --manifest resources/tasks/generated/benchmark.jsonl \
  --video-root /path/to/semlex \
  --field Contact \
  --load-4bit \
  --output experiments/results/contact_base.jsonl

python -m experiments.run \
  --manifest resources/tasks/generated/benchmark.jsonl \
  --video-root /path/to/semlex \
  --field Contact \
  --adapter experiments/results/adapters/contact \
  --load-4bit \
  --output experiments/results/contact_adapter.jsonl

python -m experiments.analyze \
  --results experiments/results/contact_adapter.jsonl \
  --compare experiments/results/contact_base.jsonl
```

The comparison aligns instance IDs and checks that the questions, videos and
answer choices agree. It reports accuracy change and the numbers of corrected
and harmed predictions.

## 6. Frozen visual probes

Extract one mean-pooled projected visual embedding per available train/test
video, then fit the field classifiers:

```bash
python -m src.qwen.extract \
  --video-root /path/to/semlex \
  --output experiments/results/embeddings

python -m experiments.probe \
  --embeddings experiments/results/embeddings \
  --output experiments/results/probes
```

Embeddings use Qwen's frozen visual encoder. Each field gets a standardized
linear classifier with class-weighted cross-entropy. Normalization is fitted
on training embeddings only. The defaults are 60 epochs, batch size 512,
learning rate `1e-3`, weight decay `1e-4` and seed 42. Cached embeddings allow
the probes to train on CPU. Use `--field Contact` to fit one classifier.

## Checks

These checks use synthetic inputs and run without downloading model weights:

```bash
python -m tests.check_graph
python -m tests.check_metrics
python -m tests.check_features
python -m tests.check_runner
```

`python -m tests.check_video` also checks decoding and visual controls after
NumPy, Pillow and OpenCV are installed.


## Citation

If you use PhonoBench in your research, please cite:

Aditya Singh Rathore, C. V. Jawahar, and Vinay P. Namboodiri.
**PhonoBench: Evaluating Sign-Form Reasoning in Video MLLMs.**
IEEE/CVF Winter Conference on Applications of Computer Vision
(WACV), 2027.

See [CITATION.cff](CITATION.cff) for citation metadata.


## References and license

Model interfaces follow the official
[Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and
[InternVL3 model card](https://huggingface.co/OpenGVLab/InternVL3-8B-hf).

**Code license:** The PhonoBench source code in this repository is released
under MIT. This license does **not** apply to external datasets or model weights.

**Dataset terms:** Sem-Lex asks users to accept its
[dataset terms of use](https://github.com/leekezar/SemLex) before downloading.
The [sign-lang@LREC catalogue](https://www.sign-lang.uni-hamburg.de/lrec/data/semlexbenchmark.html)
lists Sem-Lex as CC BY-NC-SA 4.0; consult the dataset's own terms for the
conditions that apply to your downloaded copy. The
[ASL-LEX 2.0 database](https://asl-lex.org/download.html) is licensed under
CC BY-NC 4.0 (attribution required; non-commercial use). Sem-Lex's GitHub
code license does not relicense its dataset or the underlying ASL-LEX data.

When using these data, comply with the original providers' terms and cite:

- Kezar et al., *The Sem-Lex Benchmark: Modeling ASL Signs and Their Phonemes*,
  ASSETS 2023 ([paper](https://arxiv.org/abs/2310.00196)).
- Sevcikova Sehyr et al., *The ASL-LEX 2.0 Project*,
  JDSDE 2021 ([paper](https://doi.org/10.1093/deafed/enaa038)).
- Caselli et al., *ASL-LEX: A Lexical Database for American Sign Language*,
  BRM 2017 ([paper](https://doi.org/10.3758/s13428-016-0742-0)).


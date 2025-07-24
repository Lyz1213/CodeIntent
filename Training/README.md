## Training

We provide scripts for fine-tuning Code LLMs with our reasoning-augmented dataset. The training pipeline consists of four stages:

### 1. Tokenizer Preparation

We extend the base tokenizer by adding special tokens (`<reasoning>`, `<intention>`, `<code>`, etc.) required by our training format.

\```bash
python prepare_llamatokenizer.py \
  --model <BASE_MODEL_NAME> \
  --out_dir <TOKENIZER_OUTPUT_DIR>
\```

- `<BASE_MODEL_NAME>`: HuggingFace model path (e.g., `codellama/CodeLlama-13b-Python-hf`)
- `<TOKENIZER_OUTPUT_DIR>`: Directory to save the modified tokenizer

### 2. Data Filtering and Preprocessing

We filter out invalid or overly long examples and truncate each instance to a maximum length.

\```bash
python filter_data.py \
  --json <RAW_JSON_PATH> \
  --tokenizer <TOKENIZER_OUTPUT_DIR> \
  --out_dir <FILTERED_DATA_DIR> \
  --max_len <MAX_SEQUENCE_LENGTH>
\```

- `<RAW_JSON_PATH>`: Path to raw data (e.g., `training_v3.json`)
- `<FILTERED_DATA_DIR>`: Output path for filtered JSON data
- `<MAX_SEQUENCE_LENGTH>`: Max token length for truncation (e.g., 4096)

### 3. Dataset Encoding

We verbalize each training instance into the full prompt format and encode it into Arrow format for efficient training.

\```bash
python encode_dataset.py \
  --json <FILTERED_DATA_DIR>/data.json \
  --tokenizer <TOKENIZER_OUTPUT_DIR> \
  --encode_dir <ENCODED_DATA_DIR> \
  --max_len <MAX_SEQUENCE_LENGTH>
\```

- `<ENCODED_DATA_DIR>`: Directory to store the encoded Arrow dataset

### 4. Fine-Tuning

We provide two training modes:

#### (a) PEFT Training (LoRA)

Fine-tune a small set of parameters using parameter-efficient methods.

\```bash
accelerate launch --config_file <ACCEL_CONFIG_PATH> \
  training/train_peft.py \
  --model_name <BASE_MODEL_NAME> \
  --tokenizer_dir <TOKENIZER_OUTPUT_DIR> \
  --train_dir <ENCODED_DATA_DIR>/arrow_dataset \
  --output_dir <OUTPUT_DIR> \
  --batch_tokens <TOKENS_PER_BATCH> \
  --epochs <EPOCHS> \
  --lora_r <LORA_R> \
  --lora_alpha <LORA_ALPHA> \
  --lr <LEARNING_RATE> \
  --flash_attn
\```

#### (b) Full Model Fine-Tuning

Optionally fine-tune all model parameters (recommended for smaller models).

\```bash
accelerate launch --config_file <ACCEL_CONFIG_PATH> \
  --main_process_port 25001 \
  training/train_full.py \
  --model_name <BASE_MODEL_NAME> \
  --tokenizer_dir <TOKENIZER_OUTPUT_DIR> \
  --train_dir <ENCODED_DATA_DIR>/arrow_dataset \
  --output_dir <OUTPUT_DIR> \
  --batch_tokens <TOKENS_PER_BATCH> \
  --epochs <EPOCHS> \
  --adam8bit \
  --flash_attn
\```

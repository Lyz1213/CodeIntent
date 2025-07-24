## Inference and Evaluation

We provide scripts for running inference and evaluating model performance on function-level code completion. This section outlines the complete pipeline from generation to metric evaluation.

### 1. Inference

We use `inference.py` to generate completions for each test instance. The generation can be configured with decoding strategy, stop sequences, and model path.

```bash
python inference.py \
    --output_dir ./results/<RESULT_DIR> \
    --level <LEVEL_NAME> \
    --moda <DECODE_MODE> \
    --model <MODEL_PATH> \
    --stop "</code>"
```

- `<MODEL_PATH>`: Path to your fine-tuned model checkpoint.
- `<LEVEL_NAME>`: Evaluation level (e.g., `test3_1` for DevEval).
- `<DECODE_MODE>`: Decoding mode (e.g., `greedy`, `temperature`, `beam`).
- `<RESULT_DIR>`: Directory to save inference outputs.

### 2. Parse Generated Code

After inference, we extract the generated code for evaluation.

```bash
python parse_results.py \
    --input ./results/<RESULT_DIR> \
    --output ./results/<RESULT_DIR> \
    --format all
```

### 3. Reference-based Evaluation

We use two reference-based metrics:

- **CodeBLEU**: Structural similarity.
- **EditSim**: Normalized edit distance.

First, run post-processing to clean and normalize predictions:

```bash
python eval/post_process.py \
    --data_path ./results/<RESULT_DIR>/parsed_results.jsonl \
    --parsed
```

Then compute CodeBLEU:

```bash
python eval/CodeBLEU/calc_code_bleu.py \
    --data_path ./results/<RESULT_DIR>/post_parsed_results.jsonl \
    --work generation \
    --lang python \
    --output_dir ./results/<RESULT_DIR>
```

### 4. Execution-based Evaluation

To compute execution-based accuracy (Pass@1), we compare generated code against test cases using the DevEval source code:

```bash
# Unpack source code used for testing
mkdir -p ./results/<RESULT_DIR>_source

tar -xzvf ./Source_Code.tar.gz -C ./results/<RESULT_DIR>_source
python ./check_source_code.py ./results/<RESULT_DIR>_source/Source_Code
```

Then run execution evaluation:

```bash
python pass_k.py \
    --output_file ./results/<RESULT_DIR>/processed_parsed_results.jsonl \
    --log_file ./results/<RESULT_DIR>/test_output.jsonl \
    --source_code_root ./results/<RESULT_DIR>_source/Source_Code \
    --data_file ./data.jsonl \
    --n 1 \
    --k 1

# Clean up
rm -rf ./results/<RESULT_DIR>_source
```

> 🔗 The evaluation source code and test data can be found at: [https://github.com/seketeam/DevEval/](https://github.com/seketeam/DevEval/)

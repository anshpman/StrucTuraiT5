# StrucTuraiT5

## Description

This project aims to automatically generate plausible hierarchical section outlines (like a table of contents) for Wikipedia articles. Given an article title and an optional short description, the underlying model suggests a sequence of section headings, including potential nesting levels, based on patterns learned from existing Wikipedia articles.

The goal is to provide a helpful starting point for structuring content, inspired by the organizational conventions developed by the Wikipedia community. This was developed as an exploration into sequence-to-sequence modeling, fine-tuning large language models efficiently, and building interactive ML applications.
## Features

* Generates suggested hierarchical section outlines for Wikipedia-style articles.
* Accepts article title and an optional short description as input to provide context.
* Utilizes a `t5-base` model fine-tuned using the **QLoRA** (Quantized Low-Rank Adaptation) technique for memory-efficient training and inference.
* Trained on a filtered dataset (~193k examples) derived from parsed English Wikipedia articles (using the Wikimedia Enterprise Dataset on Kaggle).
* Outputs the suggested structure with explicit nesting levels, formatted as `(level: Title)`.

## Screenshots


Here's a glimpse of the Streamlit application interface and examples of the generated section outlines using the fine-tuned T5-base QLoRA model.

---

**Example Outputs:**

**1. Marie Curie (Biography):**
![Marie Curie Output](https://github.com/anshpman/StrucTuraiT5/blob/5ec75778d46147caa64d3fba874791bc8d2f54d9/marie%20curie.png)
> Shows the typical, well-structured output for biographical articles.

**2. 1999 Cricket World Cup (Specific Event):**
![Cricket WC Output](https://github.com/anshpman/StrucTuraiT5/blob/5ec75778d46147caa64d3fba874791bc8d2f54d9/cricket.png)
> Demonstrates the model successfully generating a hierarchical structure with Level 1 and Level 2 sections.

**Example 3: History of Mumbai (Place/History):**
![History of Mumbai Output](https://github.com/anshpman/StrucTuraiT5/blob/5ec75778d46147caa64d3fba874791bc8d2f54d9/mumbai.png) 
> Shows typical top-level sections generated for the history of a major city.

**Example 4: Neural Dust (Technical Concept):**
![Neural Dust Output](neural%20dust.png)
> Shows a relevant, non-repetitive structure generated for a specific technical concept.

---

## Tech Stack

* **Programming Language:** Python 3
* **Main ML Framework:** PyTorch
* **Core Libraries:**
    * Hugging Face `transformers`: For base model (`t5-base`), tokenizer, `Seq2SeqTrainer`.
    * Hugging Face `peft`: For QLoRA implementation (`LoraConfig`, `get_peft_model`).
    * Hugging Face `datasets`: For handling data during training.
    * `bitsandbytes`: For 4-bit model quantization.
    * `accelerate`: For optimizing training and loading.
    * `pandas`: For data manipulation and filtering.
    * `streamlit`: For building the interactive web UI.
    * `sentencepiece`, `protobuf`: Dependencies for T5.
* **Base Model:** `t5-base` (from Hugging Face Hub)
* **Development Environment:**
    * Kaggle Notebooks (with GPU - Tesla T4) for data processing and model fine-tuning.
    * Local environment (e.g., VS Code) for Streamlit app development.
## Project Architecture / Workflow

The project follows the pipeline illustrated below, moving from raw data processing and model training on Kaggle to inference within a local Streamlit application:

![Project Flowchart](https://github.com/anshpman/StrucTuraiT5/blob/5ec75778d46147caa64d3fba874791bc8d2f54d9/group%2017%20better.png) 

**Explanation:**

1.  **Kaggle Dataset:** The process starts with the raw `.jsonl` article data from the Kaggle Wikipedia Structured Contents dataset.
2.  **Data Prep & Filter:** This stage (run on Kaggle, covering Modules 1-3.5) takes the raw data, parses it, extracts hierarchical structures, formats the input/output text, combines data from multiple source files, and filters out simpler articles. The output is the filtered V4 training and validation CSV files.
3.  **QLoRA Training:** Using the V4 filtered data and the pre-trained `t5-base` **Base Model** (from Hugging Face Hub), this stage (run on Kaggle, Module 4) fine-tunes the model using the QLoRA technique with the PyTorch Trainer. Only the adapter weights are trained.
4.  **PEFT Adapter:** The result of the training is the set of fine-tuned **PEFT Adapter** weights (specifically V4 QLoRA), which are saved along with the necessary configuration and tokenizer files.
5.  **Streamlit App:** This local application (Module 6):
    * Takes **User Input** (article title and description).
    * Loads the **Base Model** (`t5-base`, likely 8-bit quantized for efficiency).
    * Loads the trained **PEFT Adapter** files.
    * Combines them to perform inference, generating the section structure string.
    * Parses the string and displays the hierarchical structure as the **User Output** in the UI.
## Model

This project utilizes a pre-trained Transformer model fine-tuned for the specific task of generating Wikipedia section structures.

* **Base Model:** The foundation is **`t5-base`**, a powerful and widely used encoder-decoder model developed by Google, accessed via the Hugging Face Hub. Its text-to-text architecture is well-suited for sequence generation tasks like this.
    * You can find the base model here: [t5-base on Hugging Face Hub](https://huggingface.co/t5-base)

* **Fine-tuning Technique:** Standard fine-tuning of `t5-base` proved too memory-intensive for readily available hardware (like Kaggle's 16GB GPUs). Therefore, we employed **QLoRA (Quantized Low-Rank Adaptation)** using the Hugging Face `peft` library:
    * **Quantization:** The base `t5-base` model was loaded using 4-bit quantization via the `bitsandbytes` library. This significantly reduces the model's memory footprint during loading and training.
    * **LoRA (Low-Rank Adaptation):** Small, trainable adapter layers were added to the quantized base model (specifically targeting the query `q` and value `v` matrices in the attention mechanism, with rank `r=16` and `alpha=32`). The original base model weights remained frozen.
    * **Result:** Only these lightweight LoRA adapter weights (~1.7 million parameters, <1% of the base model) were updated during training, making the process feasible on limited hardware while still leveraging the power of `t5-base`.

* **Training:**
    * The QLoRA model was trained for **3 epochs** on the filtered V4 dataset (~193k examples).
    * Training was performed using the PyTorch-based Hugging Face `Trainer` API with mixed precision (`fp16`) enabled on a Kaggle Tesla T4 GPU.
    * The final saved artifact contains the **trained LoRA adapter weights and configuration**, not the full base model.

* **Inference:**
    * To use the model, the base `t5-base` model must be loaded first (e.g., 8-bit quantized), and then the trained PEFT adapter weights are loaded on top using the `PeftModel` class.
## Dataset

This project utilizes the **Wikipedia Structured Contents** dataset, an early beta release available on Kaggle and provided by Wikimedia Enterprise. This dataset offers pre-parsed English and French Wikipedia articles, including infoboxes and section data, in a structured JSONL format.

* **Source:** [Wikipedia Structured Contents | Kaggle](https://www.kaggle.com/datasets/wikimedia/wikipedia-structured-data)
* **Data Used:** We used the first two files containing English articles from the main namespace:
    * `enwiki_namespace_0_0.jsonl`
    * `enwiki_namespace_0_1.jsonl`
* **Processing Pipeline:**
    1.  **Parsing:** Raw JSONL lines were parsed into dictionaries.
    2.  **Feature Extraction:** Article `name` and `description` were extracted for the model input.
    3.  **Hierarchy Extraction:** The nested `sections` data within each article's JSON was recursively processed to generate a list of `(level, title)` tuples representing the article's structure.
    4.  **Formatting:** Input was formatted as `Name [SEP] Description`. The target hierarchy was formatted into a single string like `(1: Title A) | (2: Subtitle B) | ...`.
    5.  **Combining:** Data from the two source files was combined (~374k articles processed initially).
    6.  **Filtering:** The combined dataset was analyzed based on the extracted structure. To focus on articles with more substance, articles with fewer than 3 sections were filtered out, resulting in the final V4 dataset of ~193k examples used for training.
    7.  **Splitting:** The filtered V4 dataset was split into 80% training and 20% validation sets.

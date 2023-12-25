# LanguageModel-HTML-CodeGen
### This is the fineTuned model hosted on Hugging Face Hub
#### https://huggingface.co/MG650/CodeLlama_HTML_FineTuned

### Files and theie uses

#### Data_Loading and FineTuning.ipynb
1. Import the necessary libraries
2. Dataset Loading:
  Using load_dataset from the datasets library to load a dataset named "jawerty/html_dataset" from Hugging Face Hub.
  Splits the dataset into training and testing sets, with 80% for training and 20% for testing.
3. Model and Tokenizer Setup:
  model_id: Identifier for a pre-trained model, "TinyPixel/CodeLlama-7B-Instruct-bf16-sharded".
  quantization_config: Configures the model to use 4-bit quantization for reduced memory usage and potentially faster computation.
  tokenizer: Loads a tokenizer corresponding to the model. The tokenizer converts text into a format the model can understand.
  model: Loads the actual model with the specified quantization configuration.
4. Model Preparation:
  The model is prepared for training or inference, with adjustments for using 4-bit quantization.
  The tokenizer's padding token is set to its end-of-string token, if available.
5. Tokenize Function:
  A function named tokenize_function is defined to tokenize the dataset.
  It tokenizes the inputs (assumed to be in the 'label' field of the dataset) and the targets (assumed to be in the 'html' field) using the previously loaded tokenizer.
  The tokenized inputs and labels are truncated or padded to a maximum length of 512 tokens.
  The labels (targets) are then added to the inputs under the key 'labels'.
6. Applying Tokenization:
  The tokenize_function is applied to both the training and testing datasets using the map function, which processes the datasets in batches for efficiency.
7. Removing Unneeded Columns:
  Columns 'html' and 'label' are removed from both tokenized datasets as they are no longer needed after tokenization.
8. Setting Dataset Format:
  The datasets are set to the "torch" format, preparing them to be used with PyTorch for model training or evaluation.
9. DataLoaders:
   It creates DataLoader instances for both training and evaluation datasets, which are used to efficiently load data in batches during training and evaluation.
10. Optimizer:
    An AdamW optimizer is initialized for the model's parameters with a specified learning rate.
11. Learning Rate Scheduler:
    A linear learning rate scheduler is set up, which adjusts the learning rate over training steps.
12. Device Setup:
    The script checks for GPU availability and sets the model to run on GPU if available, otherwise on CPU. This is for efficient training and evaluation.
13. Gradient Checkpointing and PEFT Preparation:
  Enables gradient checkpointing in the model to save memory during training.
  Prepares the model for k-bit training using prepare_model_for_kbit_training for efficient training of large models.
14. Print Trainable Parameters Function:
  Defines a function to print the number of trainable parameters in the model, giving an insight into the model's size and complexity.
15. LoRA Configuration and Model Adaptation:
  Sets up a configuration for Low-Rank Adaptation (LoRA) to enhance the model's ability to fine-tune efficiently.
  Applies LoRA to the model using get_peft_model, targeting specific modules within the model.
16. Accelerate with FSDP (Fully Sharded Data Parallel) Plugin:
  Initializes the Fully Sharded Data Parallel plugin from accelerate library, which shards the model's parameters across multiple GPUs for efficient parallel training.
  Offloads state dictionary to CPU and configures it for all ranks.
  Prepares the model with Accelerator for efficient parallel training using the FSDP strategy.
17. Training using native Pytorch
  The tensor pairs are also converted to cuda to match the model's environment using to.(device) argument
  Backpropagation technique with negative gradient is being used
18. Model Evaluation:
  Iterates over the evaluation dataset without computing gradients (to save memory and compute resources).
19. Saving the finetuned model:
  Saved all the model files in a folder and zipped it to download and upload in hugging Face Hub

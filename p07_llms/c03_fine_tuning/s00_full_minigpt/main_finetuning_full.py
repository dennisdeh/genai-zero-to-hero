import torch
import datetime
import tokenizers
import os

# 0: Set global settings and load pre-trained (and fine-tuned) models
training = False
tokeniser = "custom_wiki_bpe_32k_"  # tiktoken or name of a custom tokeniser
path_custom_tokeniser = "p07_llms/c00_gpt_like_models/s01_minigpt/trained_tokenisers"
load_pt_model = "model_20260129_071023.bin"
load_ft_model = "model_20260129_071023.bin"
path_model = "p07_llms/c00_gpt_like_models/s01_minigpt/trained_models"
# get the device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use several GPUs if available
use_multi_gpu = True
# set block size for the model (consistency with data will be validated)
block_size = 512

# set tokeniser
if tokeniser == "tiktoken":
    import tiktoken

    tokeniser = tiktoken.get_encoding("cl100k_base")
else:
    from p07_llms.c00_gpt_like_models.s01_minigpt.data_collection.tokeniser import (
        load_custom_tokeniser,
    )

    tokeniser = load_custom_tokeniser(name=tokeniser, path=path_custom_tokeniser)
    # if the separation token does not exist, add it
    if tokeniser.token_to_id("<|separation|>") is None:
        tokeniser.add_special_tokens(["<|separation|>"])


# Initialize encoding and get EOT token
if hasattr(tokeniser, "encode_ordinary"):
    eot_token = tokeniser.encode_ordinary("<|endoftext|>")[0]  # tiktoken style
    sep_token = tokeniser.encode_ordinary("<|separation|>")
else:
    eot_token = tokeniser.token_to_id("<|endoftext|>")  # Hugging Face style
    sep_token = tokeniser.token_to_id("<|separation|>")

if training:
    # %% Import modules
    from p07_llms.c00_gpt_like_models.s01_minigpt.data_collection.data import (
        data_preparation_finance,
    )
    from p07_llms.c00_gpt_like_models.s01_minigpt.data_collection import (
        data_collection as dc,
    )
    from p07_llms.c00_gpt_like_models.s01_minigpt.mingpt.trainer import (
        Trainer,
        estimate_loss,
        batch_end_callback,
    )
    from p07_llms.c00_gpt_like_models.s01_minigpt.mingpt.model import GPT
    import gc
    import mlflow

    # set up mlflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # %% 1: Data collection and preparation
    # use the curated finance QnA dataset as an example, which is prepared in step1_data_collection.data
    d_qna = dc.dc_finance_qna(n_samples=None, threshold_skip_long_text=1000)

    # get dataset
    dataset_train, dataset_val = data_preparation_finance(
        d_qna=d_qna,
        tokeniser=tokeniser,
        block_size=block_size,
    )
    vocab_size = int(
        tokeniser.n_vocab
        if hasattr(tokeniser, "n_vocab")
        else tokeniser.get_vocab_size()
    )
    print(
        f"Data preparation complete:\n"
        f"   tokeniser encoding object: {tokeniser.__class__.__name__}\n"
        f"   vocab size: {vocab_size}\n"
        f"   block size: {block_size}\n"
        f"   number of training examples: {len(dataset_train)}\n"
        f"   number of validation examples: {len(dataset_val)}"
    )

    # %% 3: Model fine-tuning
    print("\n\n *** 3. Model fine-tuning (full) *** ")
    # instantiate the model class and load parameters from the pre-training stage
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = 32000
    model_config.block_size = block_size
    model = GPT(model_config)
    try:
        # Attempt to load the weights
        state_dict = torch.load(
            os.path.join(path_model, load_pt_model),
            map_location=device,
        )
        model.load_state_dict(state_dict)
        print("Successfully loaded weights from the saved model")
    except FileNotFoundError:
        print("Model file not found")
    except Exception as e:
        print(f"Error loading model: {e}")

    if use_multi_gpu:
        # required if several GPUs are to be used available
        model = torch.nn.DataParallel(model, device_ids=None)
    model = model.to(device)

    # Start MLflow run
    mlflow.set_experiment("minGPT-FineTuning")
    mlflow.start_run()

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 1e-6
    train_config.max_iters = 5000
    train_config.batch_size = 32 if use_multi_gpu else 16
    train_config.betas = (0.9, 0.999)
    train_config.weight_decay = 0.01  # only applied on matmul weights
    train_config.grad_norm_clip = 1.0
    # early stopping parameters
    train_config.early_stopping_rounds = 10  # *250
    train_config.num_workers = 4
    trainer = Trainer(train_config, model, dataset_train)

    # Log hyperparameters to MLflow
    mlflow.log_params(
        {
            "model_type": model_config.model_type,
            "tokenizer": tokeniser.__class__.__name__,
            "block_size": block_size,
            "learning_rate": train_config.learning_rate,
            "max_iters": train_config.max_iters,
            "batch_size": train_config.batch_size,
            "vocab_size": vocab_size,
        }
    )

    # define a callback to evaluate the loss on the validation set
    def eval_callback(trainer):
        loss = batch_end_callback(trainer, dataset_val=dataset_val, device=device)
        mlflow.log_metric("train_loss", trainer.loss.item(), step=trainer.iter_num)
        if loss is not None:
            mlflow.log_metric("val_loss", loss, step=trainer.iter_num)
        return loss

    # set callbacks and clear caches
    trainer.set_callback("on_batch_end", eval_callback)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # releases cached memory back to CUDA driver
        torch.cuda.ipc_collect()
    # run training loop
    trainer.run()

    # Final evaluation
    final_val_loss = estimate_loss(model, dataset_val, device, eval_iters=100)
    print(f"Final validation loss after training: {final_val_loss:.5f}")
    mlflow.log_metric("final_val_loss", final_val_loss)

    # save the model
    # Use .module.state_dict() if wrapped to avoid prefixing keys with "module."
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        raw_model.state_dict(),
        os.path.join(
            path_model,
            f"model_ft_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bin",
        ),
    )
    # do some simple inference
    prompt = "How do I make money?<|separation|>"
    if isinstance(tokeniser, tokenizers.Tokenizer):
        tokens = tokeniser.encode(prompt).ids
    else:
        tokens = tokeniser.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    output_ids = raw_model.generate(input_ids, max_new_tokens=50)
    print(tokeniser.decode(output_ids[0][len(tokens) - 1 :].tolist()))

    mlflow.end_run()

else:
    # %% Inference from a pre-trained model
    from p07_llms.c00_gpt_like_models.s01_minigpt.mingpt.model import GPT

    # Initialize encoding and get EOT token
    if hasattr(tokeniser, "encode_ordinary"):
        eot_token = tokeniser.encode_ordinary("<|endoftext|>")[0]  # tiktoken style
    else:
        eot_token = tokeniser.token_to_id("<|endoftext|>")  # Hugging Face style

    # load model
    # instantiate the model
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = 32000
    model_config.block_size = block_size
    model = GPT(model_config)
    try:
        # Attempt to load the weights
        state_dict = torch.load(
            os.path.join(path_model, load_ft_model),
            map_location=device,
        )
        model.load_state_dict(state_dict)
        print("Successfully loaded weights from the saved model")
    except FileNotFoundError:
        print("Model file not found")
    except Exception as e:
        print(f"Error loading model: {e}")

    model.to(device)

    # do some simple inference
    prompt = "He then went to the beach"
    if isinstance(tokeniser, tokenizers.Tokenizer):
        tokens = tokeniser.encode(prompt).ids
    else:
        tokens = tokeniser.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Manual generation loop to allow early stopping
    max_new_tokens = 50
    for _ in range(max_new_tokens):
        # crop context if it exceeds block size
        input_cond = (
            input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(input_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :]
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        next_id = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence
        input_ids = torch.cat((input_ids, next_id), dim=1)

        # STOP if we generated the EOT token
        if next_id.item() == eot_token:
            break

    # Decode and print the result (excluding the initial prompt)
    generated_tokens = input_ids[0][len(tokens) :].tolist()
    print(prompt + tokeniser.decode(generated_tokens))

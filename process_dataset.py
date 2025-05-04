from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os
import random
import numpy as np
import torch


def exact_multi_round_prompt(dataset, add_response_tokens=None):
    df = dataset.to_pandas()
    ans_df = pd.DataFrame(columns=['prompt', 'labels', 'conversation_id', 'turn_id'])

    n_illegal_samples = 0
    for conversation_id in range(len(df)):
        if conversation_id % 10000 == 0:
            print('Processing conversation ' + str(conversation_id))
        sample = df.iloc[conversation_id]
        conversation = sample['conversation']
        dialogue_so_far = ''

        new_samples = {'prompt': [],
                       'labels': [],
                       'conversation_id': [],
                       'turn_id': []}

        for i, sentence in enumerate(conversation):
            if sentence['role'] == 'user':
                dialogue_so_far += '[USER]: ' + sentence['content'] + '\n'
            else:
                assistant_content = sentence['content']

                encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
                # Drop abnormal samples that have empty responses or might have been truncated.
                if len(encoded_response['input_ids']) <= 1 or len(encoded_response['input_ids']) >= 512:
                    break

                prompt_with_preview = dialogue_so_far
                # Add specified number of words from the response if requested
                tokens_to_add = add_response_tokens if add_response_tokens is not None else ADD_RESPONSE_TOKENS
                if tokens_to_add > 0:
                    response_words = assistant_content.split()
                    preview_words = ' '.join(response_words[:min(tokens_to_add, len(response_words))])
                    prompt_with_preview += preview_words + '\n'

                # Add a new prediction sample
                new_samples['prompt'].append(prompt_with_preview)
                new_samples['conversation_id'].append(conversation_id)
                new_samples['turn_id'].append(i // 2)
                new_samples['labels'].append(len(encoded_response['input_ids']))
                dialogue_so_far += '[ASSISTANT]: ' + sentence['content'] + '\n'

        new_samples = pd.DataFrame(new_samples)
        ans_df = pd.concat([ans_df, new_samples], ignore_index=True)

    ans_dataset = Dataset.from_pandas(ans_df)
    print('Number of illegal samples: ', n_illegal_samples)
    return ans_dataset


def extract_first_round_prompt(example, add_response_tokens=None):
    conversation = example['conversation']
    user_content = ''

    # Combining the sentences from the first-round of the user prompt
    for i, sentence in enumerate(conversation):
        if sentence['role'] == 'user':
            if i > 0:
                user_content += '\n'
            user_content += sentence['content']
        else:
            break

    # Combining the sentences from the first-round of the assistant response
    assistant_content = ''
    for j in range(i, len(conversation)):
        sentence = conversation[j]
        if sentence['role'] == 'assistant':
            if j > i:
                assistant_content += '\n'
            assistant_content += conversation[j]['content']
        else:
            break

    # Add specified number of words from the response without marker
    tokens_to_add = add_response_tokens if add_response_tokens is not None else ADD_RESPONSE_TOKENS
    if tokens_to_add > 0 and assistant_content:
        response_words = assistant_content.split()
        preview_words = ' '.join(response_words[:min(tokens_to_add, len(response_words))])
        example['prompt'] = user_content + '\n' + preview_words
    else:
        example['prompt'] = user_content

    encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
    example['labels'] = len(encoded_response['input_ids'])
    return example


def tokenize_function(example):
    example = bert_tokenizer(example["prompt"], truncation=False)
    if len(example['input_ids']) >= 512:
        example['input_ids'] = example['input_ids'][-512:]
        example['token_type_ids'] = example['token_type_ids'][-512:]
        example['attention_mask'] = example['attention_mask'][-512:]
    return example


def preprocess_dataset(dataset, add_response_tokens=None):
    dataset = dataset.remove_columns(['openai_moderation', 'redacted', 'language', 'conversation_id', 'turn', 'model'])
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('prompt', new_sentence_column)
    new_label_column = [0] * len(dataset)
    dataset = dataset.add_column('labels', new_label_column)

    # Extract the user prompt(s) and the corresponding response length
    if add_response_tokens is not None:
        dataset = dataset.map(lambda x: extract_first_round_prompt(x, add_response_tokens),
                              remove_columns=['conversation'])
    else:
        dataset = dataset.map(extract_first_round_prompt, remove_columns=['conversation'])

    print('Num samples before filtering: ', len(dataset))
    dataset = dataset.filter(lambda example: example["labels"] > 1 and example["labels"] <= 512)
    print('Num samples after filtering: ', len(dataset))

    # Modified to NOT remove the 'prompt' column when tokenizing
    dataset = dataset.map(tokenize_function, batched=False)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=1000)
    parser.add_argument('--response_tokens_list', type=str, default="0",
                        help='Comma-separated list of token counts to generate datasets for (e.g., "0,1,3,5,10")')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Only vicuna-13b model
    dataset_name = 'lmsys/lmsys-chat-1m'
    model_name = 'bert-base-uncased'
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    selected_data_size = 1000 * args.data_size

    # Load the dataset once
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(selected_data_size))
    # Filter to only include vicuna-13b samples
    dataset = dataset.filter(lambda example: example["model"] == "vicuna-13b")
    dataset = dataset.shuffle(seed=args.seed)

    # Split the dataset BEFORE preprocessing to prevent data leakage
    train_test_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_raw = train_test_split['train']
    temp_eval_raw = train_test_split['test']

    # Further split the test set into validation and test
    val_test_split = temp_eval_raw.train_test_split(test_size=0.5, seed=args.seed)
    val_raw = val_test_split['train']
    test_raw = val_test_split['test']

    print(f"Raw dataset split sizes: Train={len(train_raw)}, Validation={len(val_raw)}, Test={len(test_raw)}")

    # Create directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Parse the list of response token counts
    try:
        token_counts = [int(count) for count in args.response_tokens_list.split(',')]
        print(f"Generating datasets with the following response token counts: {token_counts}")
    except ValueError:
        print(f"Error parsing response token list. Please provide comma-separated integers.")
        exit(1)

    # Process dataset once for each token count in the list
    for token_count in token_counts:
        print(f"\n=== Processing dataset with {token_count} response tokens ===\n")

        # Initialize ADD_RESPONSE_TOKENS for this iteration
        ADD_RESPONSE_TOKENS = token_count

        # Create dataset path with information about response tokens
        dataset_path = 'vicuna-13b_'

        # Add information about response preview tokens to the path name
        if token_count > 0:
            dataset_path += f'preview{token_count}_'

        dataset_path = 'data/lmsys_' + dataset_path + f'{int(selected_data_size / 1000)}K'

        # Process each split SEPARATELY with appropriate token settings
        # Process training data with requested token count
        print(f"Processing training data with {token_count} response tokens...")
        train_dataset = preprocess_dataset(train_raw, add_response_tokens=token_count)
        train_dataset.set_format("torch")

        # Process validation data with the same response tokens as training
        print(f"Processing validation data with {token_count} response tokens...")
        val_dataset = preprocess_dataset(val_raw, add_response_tokens=token_count)
        val_dataset.set_format("torch")

        # Process test data with ZERO response tokens regardless of training setting
        print(f"Processing test data with 0 response tokens (for proper evaluation)...")
        test_dataset = preprocess_dataset(test_raw, add_response_tokens=token_count)
        test_dataset.set_format("torch")

        print(
            f"Processed dataset split sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

        base_path = dataset_path.rstrip('/')

        # Save the datasets to disk
        train_dataset.save_to_disk(f"{base_path}_train")
        val_dataset.save_to_disk(f"{base_path}_val")
        test_dataset.save_to_disk(f"{base_path}_test")

        print(f'Saved train dataset to {base_path}_train')
        print(f'Saved validation dataset to {base_path}_val')
        print(f'Saved test dataset to {base_path}_test')

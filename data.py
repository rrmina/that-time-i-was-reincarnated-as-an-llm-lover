import torch
import spacy
import datasets as hf_datasets
from pathlib import Path
import pickle

from copy import copy
from collections import Counter
from typing import List, Tuple, Dict, Any

# Define source and target languages
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'de'

# Define special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

#####################################################################
#
#                    TranslationDataset class
#
#####################################################################
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_name: str = 'bentrevett/multi30k',
        split: str = 'train',
        use_cache: bool = True
    ) -> None:
        super().__init__()
    
        self.dataset_name = dataset_name
        self.split = split
        self.use_cache = use_cache
        
        # Setup cache directory
        self.cache_dir = Path('data') / f"{dataset_name.replace('/', '_')}_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset from HuggingFace repository
        print('Downloading and extracting dataset...')
        self.train_data, self.test_data, self.validation_data = self._download_and_extract_dataset()
        self.data_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'validation': self.validation_data
        }
        
        # Load tokenizer from spacy
        print('Loading tokenizers...')
        self.src_tokenizer, self.trg_tokenizer = self._load_tokenizer()
        self.tokenizers = {
            SRC_LANGUAGE: self.src_tokenizer,
            TRG_LANGUAGE: self.trg_tokenizer
        }
    
        # Build or load vocabularies
        self.src_vocab, self.trg_vocab = self._get_or_build_vocabularies()
        self.vocabularies = {
            SRC_LANGUAGE: self.src_vocab,
            TRG_LANGUAGE: self.trg_vocab
        }
        
        # Tokenize or load cached tokenized data
        data = self._get_or_tokenize_data()
        
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = data

        print('Dataset initialization complete.')

    def get_src_vocab_size(self) -> int:
        return len(self.src_vocab)

    def get_trg_vocab_size(self) -> int:
        return len(self.trg_vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,
        idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.data[idx]
        src_tensor, trg_tensor = pair

        return src_tensor, trg_tensor 
    
    def _download_and_extract_dataset(self
    ) -> Tuple[hf_datasets.Dataset, hf_datasets.Dataset, hf_datasets.Dataset]:
        # Define local storage path
        local_dataset_path = Path('data') / self.dataset_name.replace('/', '_')

        # Helper function to download dataset from HuggingFace and save to local storage
        def _download_from_hf_and_save(
            dataset_name: str,
            local_dataset_path: Path
        ) -> hf_datasets.DatasetDict:

            # Download dataset from HuggingFace repository
            hf_dataset = hf_datasets.load_dataset(dataset_name)
            
            # Save dataset to local storage
            local_dataset_path.mkdir(parents=True, exist_ok=True)
            hf_dataset.save_to_disk(str(local_dataset_path))
            print(f'Dataset saved to local storage: {local_dataset_path}')
            
            return hf_dataset

        # Check if dataset exists locally
        if local_dataset_path.exists() and any(local_dataset_path.iterdir()):
            print(f'Loading dataset from local storage: {local_dataset_path}')
            try:
                hf_dataset = hf_datasets.load_from_disk(str(local_dataset_path))
                print('Dataset loaded from local storage successfully.')
            except (OSError, ValueError, KeyError) as e:
                print(f'Failed to load local dataset: {e}')
                print('Downloading from HuggingFace...')
                hf_dataset = _download_from_hf_and_save(
                    dataset_name=self.dataset_name, 
                    local_dataset_path=local_dataset_path
                )
        else:
            print('Local dataset not found. Downloading from HuggingFace...')
            hf_dataset = _download_from_hf_and_save(
                dataset_name=self.dataset_name, 
                local_dataset_path=local_dataset_path
            )

        # Extract train, test, and validation data
        train_data = hf_dataset['train']
        test_data = hf_dataset['test']
        validation_data = hf_dataset['validation']

        return train_data, test_data, validation_data

    def _load_tokenizer(self
    ) -> Tuple[spacy.language.Language, spacy.language.Language]:
        # Load tokenizer from spacy
        src_tokenizer = spacy.load('en_core_web_sm')
        trg_tokenizer = spacy.load('de_core_news_sm')

        return src_tokenizer, trg_tokenizer
    
    def _build_vocabularies(self
    ) -> Tuple[Dict, Dict]:
        # Iterate through training data and tokenize sentences
        # Store in counters the tokens and their frequencies for both source and target languages
        # For faster tokenization, use spacy_tokenizer.pipe() instead of tokenizing sentences one by one
        src_counter, trg_counter = Counter(), Counter()
        for pair in self.train_data:
            # Extract source and target sentences from the dataset
            src_sentence, trg_sentence = pair[SRC_LANGUAGE], pair[TRG_LANGUAGE]
            
            # Tokenize source and target sentences
            src_tokens = [tok.text.lower() for tok in self.tokenizers[SRC_LANGUAGE](src_sentence)] 
            trg_tokens = [tok.text.lower() for tok in self.tokenizers[TRG_LANGUAGE](trg_sentence)]

            # Update token counters
            src_counter.update(src_tokens)
            trg_counter.update(trg_tokens)

        # Build vocabularies from special tokens
        src_vocab = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<bos>': BOS_IDX, '<eos>': EOS_IDX}
        trg_vocab = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<bos>': BOS_IDX, '<eos>': EOS_IDX}

        # Add tokens in the counters to the vocabularies with unique indices
        for token, freq in src_counter.most_common():
            if freq < 2:  # Filter out tokens that appear less than 2 times
                continue
            if token not in src_vocab:
                src_vocab[token] = len(src_vocab)

        for token, freq in trg_counter.most_common():
            if freq < 2:  # Filter out tokens that appear less than 2 times
                continue
            if token not in trg_vocab:
                trg_vocab[token] = len(trg_vocab)

        return src_vocab, trg_vocab
    
    def _get_or_build_vocabularies(self
    ) -> Tuple[Dict, Dict]:
        # Load vocabularies from cache if available, otherwise build them
        src_vocab_path = self.cache_dir / 'src_vocab.pkl'
        trg_vocab_path = self.cache_dir / 'trg_vocab.pkl'
        
        if self.use_cache and src_vocab_path.exists() and trg_vocab_path.exists():
            print('Loading vocabularies from cache...')
            try:
                with open(src_vocab_path, 'rb') as f:
                    src_vocab = pickle.load(f)
                with open(trg_vocab_path, 'rb') as f:
                    trg_vocab = pickle.load(f)
                print(f'Vocabularies loaded from cache. Src vocab size: {len(src_vocab)}, Trg vocab size: {len(trg_vocab)}')
                return src_vocab, trg_vocab
            except Exception as e:
                print(f'Failed to load vocabularies from cache: {e}')
                print('Rebuilding vocabularies...')
        
        # Build vocabularies from scratch
        print('Building vocabularies...')
        src_vocab, trg_vocab = self._build_vocabularies()
        
        # Save to cache
        if self.use_cache:
            print('Saving vocabularies to cache...')
            try:
                with open(src_vocab_path, 'wb') as f:
                    pickle.dump(src_vocab, f)
                with open(trg_vocab_path, 'wb') as f:
                    pickle.dump(trg_vocab, f)
                print(f'Vocabularies saved to {self.cache_dir}')
            except Exception as e:
                print(f'Failed to save vocabularies to cache: {e}')
        
        return src_vocab, trg_vocab
    
    def _get_or_tokenize_data(self
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # Load tokenized data from cache if available, otherwise tokenize it
        data_cache_path = self.cache_dir / f'{self.split}_data.pt'
        
        if self.use_cache and data_cache_path.exists():
            print(f'Loading tokenized {self.split} data from cache...')
            try:
                # Use torch.load with explicit weights_only=False for our trusted cache
                data = torch.load(data_cache_path, weights_only=False)
                print(f'Loaded {len(data)} tokenized sentence pairs from cache.')
                return data
            except Exception as e:
                print(f'Failed to load tokenized data from cache: {e}')
                print('Re-tokenizing data...')
        
        # Tokenize data from scratch
        print('Tokenizing data and converting to tensors...')
        data = self._tokenize_and_convert_to_tensors()
        
        # Save to cache using torch.save for tensor data
        if self.use_cache:
            print(f'Saving tokenized {self.split} data to cache...')
            try:
                torch.save(data, data_cache_path)
                print(f'Tokenized data saved to {self.cache_dir}')
            except Exception as e:
                print(f'Failed to save tokenized data to cache: {e}')
        
        return data
    
    def _tokenize_and_convert_to_tensors(self
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        # Extract dataset split
        data_split = self.data_dict[self.split]

        # Perform tokenization and conversion to tensors for each sentence pair in the dataset split
        data = []
        for pair in data_split:
            src_sentence, trg_sentence = pair[SRC_LANGUAGE], pair[TRG_LANGUAGE]
            
            # Tokenize source and target sentences
            src_tokens = [tok.text.lower() for tok in self.tokenizers[SRC_LANGUAGE](src_sentence)]
            trg_tokens = [tok.text.lower() for tok in self.tokenizers[TRG_LANGUAGE](trg_sentence)]

            # Convert tokens to indices using vocabularies
            src_indices = [self.src_vocab.get(token, UNK_IDX) for token in src_tokens]
            trg_indices = [self.trg_vocab.get(token, UNK_IDX) for token in trg_tokens]

            # Convert to tensors and add BOS/EOS tokens
            src_tensor = torch.cat([
                torch.tensor([BOS_IDX]),
                torch.tensor(src_indices),
                torch.tensor([EOS_IDX])
            ]).long()
            
            trg_tensor = torch.cat([
                torch.tensor([BOS_IDX]),
                torch.tensor(trg_indices),
                torch.tensor([EOS_IDX])
            ]).long()

            data.append((src_tensor, trg_tensor))

        return data

####################################################################
#
#                         Bucket Sampler
#
####################################################################

# Create bucket batches from the translation dataset using the specified batch size and maximum padding length
def create_bucket_batches_from_translation_dataset(
    dataset: TranslationDataset,
    batch_size: int,
    max_pad_len: int,
    debug: bool = False
) -> List[List[int]]:

    # Create iterator for dataset and compute sentence lengths for all sentence pairs
    data_iterator = iter(dataset)
    sentence_length = [(len(src), len(trg)) for src, trg in data_iterator]

    if debug:
        print(f'Starting bucket sampling with batch_size={batch_size}, max_pad_len={max_pad_len}')
        print(f'Total sentence pairs: {len(sentence_length)}\n')

    # Sort sentence pairs by their lengths and keep track of the original indices
    sorted_indices = [
        i for _,i 
        in sorted(
            zip(
                sentence_length, 
                [j for j in range(len(sentence_length))]
            )
        )
    ]
    sentence_length = copy(sorted(sentence_length))

    # Create batches using bucket sampling
    taken = [False] * len(sentence_length)
    batch_list_indices = []
    batch_num = 0
    while sum(taken) < len(sentence_length):
        src_min, src_max, trg_min, trg_max = None, None, None, None
        batch_indices = []
        counter = 0
        batch_num += 1
        
        if debug:
            print(f'--- Batch {batch_num} ---')

        while counter < len(sentence_length):
            
            # Extract data for the current sentence pair
            src_len, trg_len = sentence_length[counter]
            idx = sorted_indices[counter]

            # Check if the current sentence pair has already been taken
            # then continue to the next sentence pair
            if taken[counter]:
                counter += 1
                continue

            # New batch if src_min and trg_min are not set
            # then continue to the next sentence pair
            if src_min is None:
                batch_indices.append(idx)
                src_min, src_max, trg_min, trg_max = src_len, src_len, trg_len, trg_len
                taken[counter] = True
                if debug:
                    print(f'  Starting batch with idx {idx}: src_len={src_len}, trg_len={trg_len}')
                counter += 1
                continue

            # Check if the current sentence pair can be added to the current batch
            # if the current sentence pair fits within the current batch, 
            # then add it to the batch and update the batch statistics
            # then continue to the next sentence pair
            c1, c2 = abs(src_len - src_min), abs(src_len - src_max)
            c3, c4 = abs(trg_len - trg_min), abs(trg_len - trg_max)
            
            if c1 <= max_pad_len and c2 <= max_pad_len and c3 <= max_pad_len and c4 <= max_pad_len:
                batch_indices.append(idx)
                src_min, src_max = min(src_min, src_len), max(src_max, src_len)
                trg_min, trg_max = min(trg_min, trg_len), max(trg_max, trg_len)
                taken[counter] = True
                if debug:
                    print(f'  Added idx {idx}: src_len={src_len}, trg_len={trg_len} (batch size: {len(batch_indices)})')
                counter += 1
            else:
                if debug:
                    print(f'  Skipped idx {idx}: src_len={src_len}, trg_len={trg_len} (exceeds padding)')
                counter += 1

            # Check if the current batch is full or no more sentence
            if len(batch_indices) == batch_size or counter >= len(sentence_length):
                batch_list_indices.append(batch_indices)
                if debug:
                    print(f'  Batch complete: {len(batch_indices)} samples, src_range=[{src_min},{src_max}], trg_range=[{trg_min},{trg_max}]')
                    print(f'  Remaining: {len(sentence_length) - sum(taken)}\n')
                break

    print(f'=== Bucket sampling complete ===')
    print(f'Total batches: {len(batch_list_indices)}')
    print(f'Total samples: {sum(len(b) for b in batch_list_indices)}')
    print(f'Average batch size: {sum(len(b) for b in batch_list_indices) / len(batch_list_indices):.3f}')

    return batch_list_indices

####################################################################
#
#                        Collate Function
#
####################################################################

# Collate function to be used in the DataLoader for batching and padding sequences
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:

    # Extract source and target tensors from the batch and compute their lengths
    src_len = []
    src_batch, trg_batch = [], []
    for (src_tensor, trg_tensor) in batch:
        src_len.append(src_tensor.size(0))
        src_batch.append(src_tensor)
        trg_batch.append(trg_tensor)

    # Sort by source length (descending)
    indices = [i for i in range(len(src_len))]
    sorted_len_idx = sorted(zip(src_len, indices), reverse=True)
    sorted_src_lens = [i for i, _ in sorted_len_idx]
    sorted_indices = [idx for _, idx in sorted_len_idx]
    src_batch = [src_batch[idx] for idx in sorted_indices]
    trg_batch = [trg_batch[idx] for idx in sorted_indices]

    # Pad sequences to the length of the longest sequence in the batch
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_IDX)

    return src_batch, sorted_src_lens, trg_batch

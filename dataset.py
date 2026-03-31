import torch
import spacy
import pickle
import datasets as hf_datasets

from pathlib import Path
from copy import copy
from collections import Counter
from typing import List, Tuple, Dict

from utils import load_pickle, save_pickle

# Define source and target languages
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'de'

# Define special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = {
    '<unk>': UNK_IDX,
    '<pad>': PAD_IDX,
    '<bos>': BOS_IDX,
    '<eos>': EOS_IDX
}

# Default tokenizer models
DEFAULT_SRC_TOKENIZER = 'en_core_web_sm'
DEFAULT_TRG_TOKENIZER = 'de_core_news_sm'

#####################################################################
#
#                    TranslationDataset class
#
#####################################################################
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_name: str = 'bentrevett/multi30k',
        split: str = 'train',
        use_cache: bool = True,
        min_freq: int = 2
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.use_cache = use_cache
        self.min_freq = min_freq

        # Setup cache directory
        self.cache_dir = Path('data') / f'{dataset_name.replace("/", "_")}_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        print('Loading dataset...')
        self.dataset = self._load_dataset()

        # Load tokenizers
        print('Loading tokenizers...')
        self.src_tokenizer, self.trg_tokenizer = self._load_tokenizers()
        self.tokenizers = {
            SRC_LANGUAGE: self.src_tokenizer,
            TRG_LANGUAGE: self.trg_tokenizer
        }

        # Build or load vocabularies
        print('Loading/building vocabularies...')
        self.src_vocab, self.trg_vocab = self._get_or_build_vocabularies()
        self.vocabularies = {
            SRC_LANGUAGE: self.src_vocab,
            TRG_LANGUAGE: self.trg_vocab
        }

        # Tokenize or load cached tokenized data
        print(f'Loading/tokenizing {self.split} split...')
        self.data = self._get_or_tokenize_data()

        print('Dataset initialization complete.')

    def get_src_vocab_size(self) -> int:
        return len(self.src_vocab)

    def get_trg_vocab_size(self) -> int:
        return len(self.trg_vocab)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    #####################################################################
    #
    #                    Dataset loading
    #
    #####################################################################
    def _load_dataset(self
    ) -> hf_datasets.DatasetDict:
        # Define local storage path
        local_dataset_path = Path('data') / self.dataset_name.replace('/', '_')

        # Check if dataset exists locally
        if local_dataset_path.exists() and any(local_dataset_path.iterdir()):
            print(f'Loading dataset from local storage: {local_dataset_path}')
            try:
                dataset = hf_datasets.load_from_disk(str(local_dataset_path))
                print('Dataset loaded from local storage successfully.')
                return dataset
            except Exception as e:
                print(f'Failed to load local dataset: {e}')
                print('Re-downloading dataset...')

        # Download dataset from HuggingFace
        print('Downloading dataset from HuggingFace...')
        dataset = hf_datasets.load_dataset(self.dataset_name)

        # Save dataset locally
        local_dataset_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(local_dataset_path))
        print(f'Dataset saved to local storage: {local_dataset_path}')

        return dataset

    #####################################################################
    #
    #                    Tokenizers
    #
    #####################################################################
    def _load_tokenizers(self
    ) -> Tuple[spacy.language.Language, spacy.language.Language]:
        src_tokenizer = spacy.load(DEFAULT_SRC_TOKENIZER)
        trg_tokenizer = spacy.load(DEFAULT_TRG_TOKENIZER)

        return src_tokenizer, trg_tokenizer


    def _tokenize_batch(self,
        sentences: List[str],
        lang: str
    ) -> List[List[str]]:
        tokenizer = self.tokenizers[lang]

        return [
            [tok.text.lower() for tok in doc]
            for doc in tokenizer.pipe(sentences, batch_size=1000)
        ]

    #####################################################################
    #
    #                    Vocabulary building
    #
    #####################################################################
    def _build_vocab(self,
        sentences: List[str],
        lang: str
    ) -> Dict[str, int]:
        # Count token frequencies using batch tokenization
        counter = Counter()
        tokenized_sentences = self._tokenize_batch(sentences, lang)

        for tokens in tokenized_sentences:
            counter.update(tokens)

        # Initialize vocabulary with special tokens
        vocab = copy(SPECIAL_TOKENS)

        # Add tokens above min frequency
        for token, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def _get_or_build_vocabularies(self
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        src_vocab_path = self.cache_dir / 'src_vocab.pkl'
        trg_vocab_path = self.cache_dir / 'trg_vocab.pkl'

        # Load vocabularies from cache if available
        if self.use_cache and src_vocab_path.exists() and trg_vocab_path.exists():
            print('Loading vocabularies from cache...')
            try:
                src_vocab = load_pickle(src_vocab_path)
                trg_vocab = load_pickle(trg_vocab_path)
                print(f'Vocabularies loaded from cache. Src vocab size: {len(src_vocab)}, Trg vocab size: {len(trg_vocab)}')
                return src_vocab, trg_vocab
            except Exception as e:
                print(f'Failed to load vocabularies from cache: {e}')
                print('Rebuilding vocabularies...')

        # Build vocabularies from training split
        print('Building vocabularies from training data...')
        train_data = self.dataset['train']

        src_sentences = [pair[SRC_LANGUAGE] for pair in train_data]
        trg_sentences = [pair[TRG_LANGUAGE] for pair in train_data]

        src_vocab = self._build_vocab(src_sentences, SRC_LANGUAGE)
        trg_vocab = self._build_vocab(trg_sentences, TRG_LANGUAGE)

        print(f'Built vocabularies. Src vocab size: {len(src_vocab)}, Trg vocab size: {len(trg_vocab)}')

        # Save vocabularies to cache
        if self.use_cache:
            print('Saving vocabularies to cache...')
            try:
                save_pickle(src_vocab, src_vocab_path)
                save_pickle(trg_vocab, trg_vocab_path)
                print(f'Vocabularies saved to {self.cache_dir}')
            except Exception as e:
                print(f'Failed to save vocabularies to cache: {e}')

        return src_vocab, trg_vocab

    #####################################################################
    #
    #                 Converting tokens to tensors
    #
    #####################################################################
    def _numericalize(self,
        tokens: List[str],
        vocab: Dict[str, int]
    ) -> torch.Tensor:
        
        indices = [vocab.get(token, UNK_IDX) for token in tokens]

        return torch.tensor([BOS_IDX] + indices + [EOS_IDX], dtype=torch.long)

    #####################################################################
    #
    #                    Tokenized data loading
    #
    #####################################################################
    def _get_or_tokenize_data(self
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data_cache_path = self.cache_dir / f'{self.split}_data.pt'

        # Load tokenized data from cache if available
        if self.use_cache and data_cache_path.exists():
            print(f'Loading tokenized {self.split} data from cache...')
            try:
                data = torch.load(data_cache_path, weights_only=False)
                print(f'Loaded {len(data)} tokenized sentence pairs from cache.')
                return data
            except Exception as e:
                print(f'Failed to load tokenized data from cache: {e}')
                print('Re-tokenizing data...')

        # Tokenize from scratch
        print(f'Tokenizing {self.split} data and converting to tensors...')
        data = self._tokenize_and_convert_to_tensors()
        print(f'✓ Tokenized {len(data)} sentence pairs.')

        # Save to cache
        if self.use_cache:
            print(f'Saving tokenized {self.split} data to cache...')
            try:
                torch.save(data, data_cache_path)
                print(f'✓ Tokenized data saved to {data_cache_path}')
            except Exception as e:
                print(f'Failed to save tokenized data to cache: {e}')

        return data

    def _tokenize_and_convert_to_tensors(self
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # Extract dataset split
        split_data = self.dataset[self.split]

        # Extract raw sentences
        src_sentences = [pair[SRC_LANGUAGE] for pair in split_data]
        trg_sentences = [pair[TRG_LANGUAGE] for pair in split_data]

        # Batch tokenize source and target
        src_tokenized = self._tokenize_batch(src_sentences, SRC_LANGUAGE)
        trg_tokenized = self._tokenize_batch(trg_sentences, TRG_LANGUAGE)

        # Convert each sentence pair into tensors
        data = []
        for src_tokens, trg_tokens in zip(src_tokenized, trg_tokenized):
            src_tensor = self._numericalize(src_tokens, self.src_vocab)
            trg_tensor = self._numericalize(trg_tokens, self.trg_vocab)

            data.append((src_tensor, trg_tensor))

        return data

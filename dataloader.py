from typing import List, Tuple
from copy import copy
from pathlib import Path

import torch

from dataset import TranslationDataset
from dataset import PAD_IDX

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
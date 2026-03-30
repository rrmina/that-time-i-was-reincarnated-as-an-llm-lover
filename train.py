# Training logic and utilities.
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.fabric import Fabric
import os

from data import TranslationDataset, create_bucket_batches_from_translation_dataset, collate_fn
from data import PAD_IDX
from model import Seq2Seq

import time

BATCH_SIZE = 124
MAX_PAD_LEN = 5
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3

ENC_EMBEDDING_DIM = ENC_HIDDEN_DIM = DEC_EMBEDDING_DIM = DEC_HIDDEN_DIM = 256

ENC_NUM_LAYERS = 3
ENC_NUM_HEADS = 8
ENC_FFN_DIM = 512

DEC_NUM_LAYERS = 3
DEC_NUM_HEADS = 8
DEC_FFN_DIM = 512

ENC_DROPOUT_PROB = DEC_DROPOUT_PROB = 0.1
ENC_MAX_SEQ_LEN = DEC_MAX_SEQ_LEN = 512


# Train the model.
def train_model(
    fabric: Fabric,
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    save_path: str = 'checkpoints/seq2seq_model'
) -> None:
    
    # TODO: Implement training loop
    # Sample implementation:

    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'checkpoints', exist_ok=True)

    # Training Loop
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        fabric.print('Epoch: [{:>2}/{:>2}]'.format(epoch + 1, num_epochs))
        net.train()
        epoch_loss = 0.0
        num_batches = 0

        # Batch loops
        for batch_idx, (src, src_lens, trg) in enumerate(train_loader):

            # Prepare data - Fabric handles device placement automatically
                                                # src: [seq_len, batch_size]
                                                # src_lens: [batch_size]
                                                # src.T: [batch_size, seq_len]
            src, trg = src.T, trg.T
            optimizer.zero_grad()

            # Seq2Seq forward pass
            #  Target input to the model is all but the last token of the target
                                                # dec_out = [batch_size, trg_seq_len, output_dim]
            dec_out = net(src, trg[:, :-1])

            # Reshape for Decoder output loss calculation
                                                # output_dim = trg_vocab_size
                                                # dec_out = [batch_size * trg_seq_len, output_dim]
            output_dim = dec_out.shape[-1]
            dec_out = dec_out.reshape(-1, output_dim)
            
            # Reshape target for loss calculation  
            #   Target is all but the first token, reshape for loss calculation  
                                                # trg = [batch_size, trg_seq_len-1]
            trg = trg[:, 1:].reshape(-1)   
        
            # Calculate loss, backpropagate, and update weights
            loss = criterion(dec_out, trg)
            fabric.backward(loss)
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1

            # Print training progress
            if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1 or batch_idx == 0:
                fabric.print('Batch: [{:>5}/{:>5}] | Loss: {:.4f}'.format(batch_idx, len(train_loader), loss.item()))
        
        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Log metrics
        fabric.log_dict({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        fabric.print(f'Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}')
        
        # Save checkpoint if best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net,
                'optimizer_state_dict': optimizer,
                'loss': avg_epoch_loss,
            }
            fabric.save(f'{save_path}_best.ckpt', checkpoint)
            fabric.print(f'Saved best model checkpoint with loss: {avg_epoch_loss:.4f}')
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net,
                'optimizer_state_dict': optimizer,
                'loss': avg_epoch_loss,
            }
            fabric.save(f'{save_path}_epoch_{epoch + 1}.ckpt', checkpoint)
            fabric.print(f'Saved checkpoint at epoch {epoch + 1}')
    
    elapsed_time = time.time() - start_time
    fabric.print(f'Training completed in {elapsed_time:.2f} seconds')
    fabric.print(f'Best loss: {best_loss:.4f}')

if __name__ == '__main__':

    # Initialize Fabric for device management and distributed training
    fabric = Fabric(accelerator='auto', devices='auto', precision='32-true')
    fabric.launch()
    
    fabric.print(f'Using device: {fabric.device}')
    fabric.print(f'World size: {fabric.world_size}')

    # Dataset and DataLoader
    trainset = TranslationDataset(
        dataset_name = 'bentrevett/multi30k',
        split = 'train'
    )
    bucket_list_indices = create_bucket_batches_from_translation_dataset(
        dataset = trainset,
        batch_size = BATCH_SIZE,
        max_pad_len = MAX_PAD_LEN,
        debug = False
    )
    trainloader = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_sampler = bucket_list_indices,
        collate_fn = collate_fn
    )

    # Get vocabulary sizes
    src_vocab_size = trainset.get_src_vocab_size()
    trg_vocab_size = trainset.get_trg_vocab_size()

    # Instantiate model
    net = Seq2Seq(
        enc_num_layers = ENC_NUM_LAYERS,
        src_vocab_size = src_vocab_size,
        enc_embedding_dim = ENC_EMBEDDING_DIM,
        enc_hidden_dim = ENC_HIDDEN_DIM,
        enc_num_heads = ENC_NUM_HEADS,
        enc_ffn_dim = ENC_FFN_DIM,

        dec_num_layers = DEC_NUM_LAYERS,
        trg_vocab_size = trg_vocab_size,
        dec_embedding_dim = DEC_EMBEDDING_DIM,
        dec_hidden_dim = DEC_HIDDEN_DIM,
        dec_num_heads = DEC_NUM_HEADS,
        dec_ffn_dim = DEC_FFN_DIM,

        enc_dropout_prob = ENC_DROPOUT_PROB,
        enc_max_seq_len = ENC_MAX_SEQ_LEN,
        dec_dropout_prob = DEC_DROPOUT_PROB,
        dec_max_seq_len = DEC_MAX_SEQ_LEN
    )

    # Set up optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Setup model, optimizer, and dataloader with Fabric
    net, optimizer = fabric.setup(net, optimizer)
    trainloader = fabric.setup_dataloaders(trainloader)

    # Train the model
    train_model(
        fabric = fabric,
        net = net,
        train_loader = trainloader,
        val_loader = None,  # TODO: Create validation loader
        optimizer = optimizer,
        criterion = criterion,
        num_epochs = NUM_EPOCHS,
        save_path = 'checkpoints/seq2seq_model'
    )


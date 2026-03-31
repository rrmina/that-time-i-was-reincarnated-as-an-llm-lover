from pathlib import Path

import spacy
import torch
from lightning.fabric import Fabric, seed_everything

from model import Seq2Seq
from utils import load_pickle

# Seed for reproducibility
SEED = 42

# Model hyperparameters
ENC_EMBEDDING_DIM = ENC_HIDDEN_DIM = DEC_EMBEDDING_DIM = DEC_HIDDEN_DIM = 256

ENC_NUM_LAYERS = 3
ENC_NUM_HEADS = 8
ENC_FFN_DIM = 512

DEC_NUM_LAYERS = 3
DEC_NUM_HEADS = 8
DEC_FFN_DIM = 512

ENC_DROPOUT_PROB = DEC_DROPOUT_PROB = 0.1
ENC_MAX_SEQ_LEN = DEC_MAX_SEQ_LEN = 512

# Default tokenizer models
DEFAULT_SRC_TOKENIZER = 'en_core_web_sm'
DEFAULT_TRG_TOKENIZER = 'de_core_news_sm'

def translate_sentence(
    net: Seq2Seq, 
    sentence: str, 
    src_vocab: dict, 
    trg_vocab_inv: dict,
    src_tokenizer: spacy.language.Language
) -> str:
    
    # Transform
    sentence = sentence.lower().strip()
    tokens = src_tokenizer(sentence)
    token_indices = [src_vocab.get(token.text, src_vocab['<unk>']) for token in tokens]

    # Add BOS and EOS to match training preprocessing
    src_tensor = torch.LongTensor(
        [src_vocab['<bos>']] + token_indices + [src_vocab['<eos>']]
    ).unsqueeze(0)

    # Move input to model device
    src_tensor = src_tensor.to(next(net.parameters()).device)

    # Generate translation
    with torch.no_grad():
        output_indices = net.translate(src_tensor)

    # Convert TRG token IDs back to words
    special_tokens = {'<bos>', '<eos>', '<pad>'}
    translation_tokens = [
        trg_vocab_inv[idx]
        for idx in output_indices
        if trg_vocab_inv[idx] not in special_tokens
    ]
    translation = ' '.join(translation_tokens)

    return translation  

def main():

    # Seed for reproducibility
    seed_everything(SEED, workers=True)

    # Enable deterministic mode for PyTorch
    torch.use_deterministic_algorithms(True, warn_only = True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_name = 'bentrevett/multi30k'
    cache_dir = Path('data') / f'{dataset_name.replace("/", "_")}_cache'
    
    # Load vocab
    src_vocab_path = cache_dir / 'src_vocab.pkl'
    trg_vocab_path = cache_dir / 'trg_vocab.pkl'
    src_vocab = load_pickle(src_vocab_path)
    trg_vocab = load_pickle(trg_vocab_path)

    trg_vocab_inv = {idx: token for token, idx in trg_vocab.items()}

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    # Load SRC tokenizer
    src_tokenizer = spacy.load(DEFAULT_SRC_TOKENIZER)

    # Load model
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

    # Setup Fabric
    fabric = Fabric(accelerator = 'auto', devices = 'auto', precision = '32-true')
    fabric.launch()
    net = fabric.setup(net)

    # Load checkpoint
    save_path = 'checkpoints/seq2seq_model_epoch_50.ckpt'
    checkpoint = fabric.load(save_path, weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])

    fabric.print('Model loaded successfully. Ready for translation!')

    net.eval()

    with torch.no_grad():
        while True:
            sentence = input("Enter an English sentence to translate (or 'exit' to quit): ")
            
            if sentence.lower() == 'exit':
                break

            translation = translate_sentence(
                net = net, 
                sentence = sentence, 
                src_vocab = src_vocab, 
                trg_vocab_inv = trg_vocab_inv, 
                src_tokenizer = src_tokenizer
            )
            
            print(f'Translation: {translation}')



if __name__ == '__main__':
    main()

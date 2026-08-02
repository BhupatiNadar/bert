from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def get_all_sentences(ds):
    for item in ds:
        text = item["text"].strip()

        if text:
            yield text


def get_or_build_tokenizer(config, ds):
    
    tokenizer_path = Path(config["tokenizer_file"])

    # Create the tokenizer directory if it does not exist
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[CLS]","[SEP]","[MASK]",],min_frequency=2,)

        tokenizer.train_from_iterator(
            get_all_sentences(ds),
            trainer=trainer,
        )

        tokenizer.save(str(tokenizer_path))

        print(f"Tokenizer created: {tokenizer_path}")

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer loaded: {tokenizer_path}")

    return tokenizer


def get_ds(config):
    """
    Load WikiText and build/load its tokenizer.
    """
    ds_raw = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
    )

    tokenizer = get_or_build_tokenizer(
        config=config,
        ds=ds_raw,
    )

    print(f"Dataset size: {len(ds_raw)}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return ds_raw, tokenizer


if __name__ == "__main__":
    config = {
        "tokenizer_file": "tokenizers/bert_tokenizer.json",
    }

    dataset, tokenizer = get_ds(config)
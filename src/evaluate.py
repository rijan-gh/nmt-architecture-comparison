# Translation, shared by all 4 models via their .translate() method
import torch


def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    """Greedy-translate one sentence string. Returns (translation_str, attention_or_None)."""
    model.eval()
    tokens = src_vocab.numericalize(sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    sos_idx = trg_vocab.word2idx[trg_vocab.SOS_TOKEN]
    eos_idx = trg_vocab.word2idx[trg_vocab.EOS_TOKEN]

    token_ids, attention = model.translate(src_tensor, sos_idx, eos_idx, max_len=max_len)
    words = [trg_vocab.idx2word.get(i, trg_vocab.UNK_TOKEN) for i in token_ids]
    return " ".join(words), attention


def compare_models(sentence, models_dict, src_vocab, trg_vocab, device, max_len=50):
    """Translate one sentence with several models at once, for a side-by-side comparison."""
    return {
        name: translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=max_len)[0]
        for name, model in models_dict.items()
    }

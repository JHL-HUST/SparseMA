import os


class InputExample(object):
    
    def __init__(self, guid, text_a, text_b=None, label=None, flaw_labels=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.flaw_labels = flaw_labels

class InputFeatures(object):
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_example_to_feature_for_bert(text, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    tokens_a = tokenizer.tokenize(text)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids

def convert_example_to_feature_for_cnn(x, map2id, max_seq_length, oov='<oov>', pad_token='<pad>'):
    input_mask = [1] * len(x) + [0] * max(max_seq_length - len(x), 0)
    oov_id = map2id[oov]
    if len(x) > max_seq_length:
        x = x[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
    else:
        x = x + [pad_token] * (max_seq_length - len(x))
    x = [map2id.get(w, oov_id) for w in x]
    # x = torch.LongTensor(x)
    return x, input_mask
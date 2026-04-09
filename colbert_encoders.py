import string
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F

Q_token = 1
D_token = 2


class ColbertEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", search_dim=128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.search_dim = search_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_query_len = 32
        self.max_doc_len = 180

        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.projection = nn.Linear(self.bert.config.hidden_size, self.search_dim, bias=False).to(self.device)

        self.punct_ids = set(
            self.tokenizer.convert_tokens_to_ids(list(string.punctuation))
        )

    def _raw_tokenizer(self, list_of_strings):
        return self.tokenizer(list_of_strings, add_special_tokens=False)

    def _add_special_query_tokens(self, tokenized):
        max_len = self.max_query_len - 3
        with_spec_tokens = [self.tokenizer.cls_token_id] + [Q_token] + tokenized[:max_len] + [self.tokenizer.sep_token_id]
        padding_len = max(0, self.max_query_len - len(with_spec_tokens))

        if padding_len > 0:
            return with_spec_tokens + [self.tokenizer.mask_token_id] * padding_len

        return with_spec_tokens

    def _tokenize_queries(self, queries):
        tokenized = self._raw_tokenizer(queries)
        input_ids = torch.tensor([self._add_special_query_tokens(t) for t in tokenized.input_ids]).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        token_type_ids = torch.zeros_like(input_ids).to(self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    def _add_special_document_tokens(self, tokenized):
        return [self.tokenizer.cls_token_id] + [D_token] + tokenized[:self.max_doc_len - 3] + [self.tokenizer.sep_token_id]

    def _tokenize_documents(self, documents):
        tokenized = self._raw_tokenizer(documents)
        ids = [self._add_special_document_tokens(t) for t in tokenized.input_ids]
        max_len = max(len(t) for t in ids)

        padded = [t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in ids]
        input_ids = torch.tensor(padded).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        token_type_ids = torch.zeros_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    def _encode(self, tokens):
        bert_output = self.bert(**tokens).last_hidden_state
        projected = self.projection(bert_output)
        normalized = F.normalize(projected, p=2, dim=-1)
        mask = tokens["attention_mask"]
        return normalized * mask.unsqueeze(-1), mask

    def encode_queries(self, queries):
        return self._encode(self._tokenize_queries(queries))

    def encode_documents(self, documents):
        tokens = self._tokenize_documents(documents)
        bert_output = self.bert(**tokens).last_hidden_state
        projected = self.projection(bert_output)
        normalized = F.normalize(projected, p=2, dim=-1)

        mask = tokens["attention_mask"].clone()
        input_ids = tokens["input_ids"]
        for punct_id in self.punct_ids:
            mask = mask & (input_ids != punct_id)

        return normalized * mask.unsqueeze(-1), mask

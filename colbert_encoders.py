import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F

Q_token = 1
D_token = 2


class ColbertEncoder(nn.Module):
    def __init__(self, encoder_branch=None, model_name="bert-base-uncased", search_dim=128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_token_len = {
            "query": 32,
            "document": 180,
        }[encoder_branch]

        self.tokenize_fn = {
            "query": self.tokenize_queries,
            "document": self.tokenize_documents,
        }[encoder_branch]

        self.search_dim = search_dim

        if encoder_branch not in ['query', 'document']:
            raise ValueError("encoder_branch must be either 'query' or 'document'")

        self.encoder_branch = encoder_branch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
      
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.projection = nn.Linear(self.bert.config.hidden_size, self.search_dim, bias=False).to(self.device)

    def _raw_tokenizer(self, list_of_strings):
        return self.tokenizer(list_of_strings, add_special_tokens=False)

    def _add_special_query_tokens(self, tokenized):
        # subtract 3 tokens for [CLS], [SEP] and Q token
        max_len = self.max_token_len - 3
        with_spec_tokens = ([self.tokenizer.cls_token_id] + [Q_token] + tokenized[:max_len] + [self.tokenizer.sep_token_id])
        padding_len = max(0, self.max_token_len - len(with_spec_tokens))

        if padding_len > 0:
            return with_spec_tokens + [self.tokenizer.mask_token_id] * padding_len
        
        return with_spec_tokens

    # procedure to tokenize queries - list of different length strings
    def tokenize_queries(self, queries):
        tokenized = self._raw_tokenizer(queries)
        input_ids = torch.tensor([self._add_special_query_tokens(t) for t in tokenized.input_ids]).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(self.device)
        token_type_ids = torch.zeros_like(input_ids).to(self.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}


    def _add_special_document_tokens(self, tokenized):
        return [self.tokenizer.cls_token_id] + [D_token] + tokenized[:self.max_token_len - 3] + [self.tokenizer.sep_token_id]

    def tokenize_documents(self, documents):
        tokenized = self._raw_tokenizer(documents)
        ids = [self._add_special_document_tokens(t) for t in tokenized.input_ids]
        max_len = max(len(t) for t in ids)

        padded = [t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in ids]
        input_ids = torch.tensor(padded).to(self.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        token_type_ids = torch.zeros_like(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    def forward(self, x):
        tokens = self.tokenize_fn(x)
        bert_output = self.bert(**tokens).last_hidden_state
        shared_space = self.projection(bert_output)
        normalized = F.normalize(shared_space, p=2, dim=-1)
        mask = tokens["attention_mask"]

        return normalized * mask.unsqueeze(-1), mask

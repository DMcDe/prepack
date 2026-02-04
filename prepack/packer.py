from abc import ABC, abstractmethod
from collections.abc import Sequence
import csv
import gzip
import json
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer
from typing import Dict, List, Iterator

class OfflineDataset(Dataset):
    """
    PyTorch Dataset representation for the original Dataset, stored as a jsonl. 
    """

    def __init__(self, documents: Sequence[str]) -> None:
        """
        Initialize an offline dataset representation.
        
        :param self: Class instance
        :param json_path: Path to the jsonl file holding the dataset.
        :type json_path: str
        :param data_col: Title of the column holding the data/text for the dataset.
        :type data_col: str
        """
        self.num_docs = len(documents)
            
    def __len__(self) -> int:
        """
        Get the length (# documents) of the dataset.
        """
        return self.num_docs
    
    def __getitem__(self, index: int) -> int:
        return index
    
class OfflinePacker(ABC):
    def __init__(self, output_csv: str, tokenizer: AutoTokenizer, documents: Sequence[str]) -> None:
        """
        Instantiate an OfflinePacker object.
        
        :param self: OfflinePacker instance.
        :param output_csv: Path & filename at which to store the CSV output of the packing algorithm.
        :type output_csv: str
        :param tokenizer: Hugging Face tokenizer to use to tokenize the dataset.
        :type tokenizer: AutoTokenizer
        :param documents: Container that provides random access to docs in the training set. Must implement __getitem__ & __len__.
        :type documents: Sequence[str]
        """
        self.output_csv = output_csv
        self.tokenizer = tokenizer
        self.documents = documents
        self.dataset = OfflineDataset(documents)

        self.file_assignments = {}

    def get_doc_lens(self, indices: List[int]) -> Dict[int, int]:
        """
        Finds the length of documents in the dataset.
        
        :param self: Description
        :param indices: Indices of documents to profile.
        :type indices: List[int]
        :return: Dict from document ID to document length, for given doc IDs.
        :rtype: Dict[int, int]
        """
        files = {}

        for i in indices:
            files[i] = len(self.tokenizer.encode(self.documents[i]))

        return files
    
    def dict_to_list(self, file_assignments: Dict[int, int]) -> List[List[int]]:
        """
        Convert a dict from doc ID to microbatch ID to a list of microbatches (each a list of IDs).
        
        :param self: OfflinePacker instance
        :param file_assignments: Dict from doc ID to microbatch assignment.
        :type file_assignments: Dict[int, int]
        :return: List of list of documents in each microbatch.
        :rtype: List[List[int]]
        """
        length = max(file_assignments.values()) + 1
        batches = [[] for _ in range(length)]

        for k, v in file_assignments.items():
            batches[v].append(k)

        # Filter out any unfilled batches
        batches = [b for b in batches if len(b) > 0]

        with open(self.output_csv, 'w') as output_csv:
            writer = csv.writer(output_csv)
            writer.writerows(batches)

        return batches
    
    @staticmethod
    def varlen_collate(batch):
        """
        Overwritten collate fn so that PyTorch allows tensors of uneven length.
        """
        return batch

    @abstractmethod
    def pack_microbatches(self, **kwargs):
        """
        Pack dataset into microbatches. Must be overriden by user.
        """
        raise NotImplementedError("Subclasses of OfflinePacker should implement pack_microbatches.")
    
class RuntimeDataset(Dataset):
    def __init__(self, batches: List[List[int]], tokenizer: AutoTokenizer, documents: Sequence[str]) -> None:
        """
        Initialize a runtime, packed dataset.
        
        :param self: RuntimeDataset instance
        :param batches: List of microbatches (each a list of doc IDs).
        :type batches: List[List[int]]
        :param tokenizer: Hugging Face tokenizer to use to tokenize the document.
        :type tokenizer: AutoTokenizer
        :param documents: Container that provides random access to docs in the training set. Must implement __getitem__ & __len__.
        :type documents: Sequence[str]
        """
        self.tokenizer = tokenizer
        self.length = sum([len(b) for b in batches])
        self.documents = documents

    def __len__(self) -> int:
        """
        Get the length (number of documents across all microbatches) of the packed dataset.
        """
        return self.length
    
    def __getitem__(self, index: int) -> Tensor:
        return self.tokenizer.encode(self.documents[index], return_tensors='pt')
    
class RuntimeBatchSampler(Sampler[List[int]]):
    """
    Custom PyTorch BatchSampler for variable lengthed microbatches, where each is a list of doc IDs.
    """
    def __init__(self, batches: List[List[int]]) -> None:
        self.batches = batches
        self.num_batches = len(batches)

    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        for b in self.batches:
            yield b

class RuntimeStreamer:
    def __init__(self, input_csv: str, documents: Sequence[str], context_window: int, n_sequences: int, tokenizer: AutoTokenizer) -> None:
        """
        Instantiate a RuntimeStreamer object.
        
        :param self: RuntimeStreamer instance
        :param input_csv: Path to the CSV file output by the offline packing stage.
        :type input_csv: str
        :param documents: Container that provides random access to docs in the training set. Must implement __getitem__ & __len__.
        :type documents: Sequence[str]
        :param context_window: Context window length for the model being trained.
        :type context_window: int
        :param n_sequences: Number of sequences per training loop.
        :type n_sequences: int
        :param tokenizer: Hugging Face tokenizer to use when tokenizing documents.
        :type tokenizer: AutoTokenizer
        """
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.context_window = context_window
        self.num_sequences = n_sequences
        self.batches = self.read_batches()
        self.sampler = RuntimeBatchSampler(self.batches)
        self.dataset = RuntimeDataset(self.batches, self.tokenizer, documents)
        self.dataloader = DataLoader(self.dataset, batch_sampler=self.sampler, collate_fn=self.collate_concat)

        # Ensure tokenizer will support combining documents when packing later
        if getattr(self.tokenizer, "eos_token", None) is None and "<|end|>" not in getattr(self.tokenizer, "all_special_tokens", []):
            raise AttributeError("Selected tokenizer doesn't have an end-of-sequence token, which is required for packing microbatches.")

    def read_batches(self) -> List[List[int]]:
        """
        Reads the CSV file output by the offline stage to get a list of microbatches.
        
        :param self: RuntimeStreamer instance
        :return: List of microbatches (each a list of doc IDs).
        :rtype: List[List[int]]
        """
        with open(self.input_csv, newline='') as input_csv:
            reader = csv.reader(input_csv)
            batches = list(reader)
            batches = [[int(s) for s in batch] for batch in batches]

        return batches
    
    def collate_concat(self, batch):
        """
        Custom collate function to generate samples from microbatches of varying lengths.
        
        :param self: RuntimeStreamer instance
        :param batch: List of tokenized documents in the given batch
        """
        # Handle case where tokenizer doesn't have EOS token
        if getattr(self.tokenizer, "eos_token", None) is not None:
            eos_token = self.tokenizer.eos_token
        elif "<|end|>" in getattr(self.tokenizer, "all_special_tokens", []):
            eos_token = "<|end|>"
        else:
            # Because of exception handling in constructor, this should never happen
            pass
        
        eos = Tensor(self.tokenizer.encode(eos_token))
        res = Tensor()
        for doc in batch:
            res = torch.cat((res, torch.flatten(doc), eos))

        # Optimization: pad to context_window_length + num_sequences + 1 & calculate step size dynamically
        # Avoids creating sequences with lots of EOS tokens
        if (res.size()[0] < self.context_window + self.num_sequences + 1):
            res = torch.cat((res, Tensor(self.tokenizer.encode(eos_token) * ((self.context_window + self.num_sequences + 1) - res.size()[0]))))

        step_size = max(1, (res.size()[0] - self.context_window) // self.num_sequences)
        
        xs = []
        ys = []
        for i in range(self.num_sequences):
            x = res[i * step_size : i * step_size + self.context_window]
            y = res[i * step_size + 1: i * step_size + 1 + self.context_window]

            xs.append(x)
            ys.append(y)

        return torch.stack(xs).long(), torch.stack(ys).long()
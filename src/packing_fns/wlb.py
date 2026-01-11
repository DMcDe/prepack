from prepack.packer import OfflinePacker
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List

class OutlierQueue:
    """
    Multilevel queue to temporarily store long documents for WLB packing.
    """

    def __init__(self, bounds: List[int]) -> None:
        """
        Initialize an outlier queue for running the WLB-LLM heuristic algorithm.
        
        :param self: OutlierQueue instance
        :param bounds: Minimum thresholds (# tokens) at which levels of the queue begin.
        :type bounds: List[int]
        """
        self.num_levels = len(bounds)
        self.outlier_queues = [[] for _ in range(self.num_levels)]
        self.bounds = bounds

    def check_outlier(self, file_name: str, file_len: int) -> bool:
        """
        Add long documents to the outlier queue.
        
        :param self: OutlierQueue instance
        :param file_name: Name of file being evaluated.
        :type file_name: str
        :param file_len: Length of file being evaluated (# tokens).
        :type file_len: int
        :return: True if doc was an outlier and got added to the queue, false otherwise.
        :rtype: bool
        """
        for i in reversed(range(self.num_levels)):
            if file_len > self.bounds[i]:
                self.outlier_queues[i].append((file_len, file_name))
                return True
            
        return False
    
class Microbatch:
    """
    Class to track the total workload and length (# tokens) of each microbatch.
    """

    def __init__(self, index: int) -> None:
        """
        Initialize a Microbatch object.
        
        :param self: Microbatch instance
        :param index: Unique numeric index for the microbatch.
        :type index: int
        """
        self.workload = 0
        self.tokens = 0
        self.index = index

    def push_document(self, tokens: int, workload: int) -> None:
        """
        Update microbatch's state to reflect a document's added length and workload.
        
        :param self: Microbatch instance
        :param tokens: Number of tokens of the added document.
        :type tokens: int
        :param workload: Workload of the added document.
        :type workload: int
        """
        self.tokens += tokens
        self.workload += workload

    @staticmethod
    def attention_workload(length: int) -> int:
        """
        Estimate the workload required for attention for a given document length.
        
        :param length: Document length (tokens)
        :type length: int
        :return: Estimated attention workload
        :rtype: int
        """
        return 1 + length**2
    
    @staticmethod
    def operation_workload(length: int) -> int:
        """
        Estimate the workload required for non-attention calculations for a given document length.
        
        :param length: Document length (tokens)
        :type length: int
        :return: Estimated operation workload
        :rtype: int
        """
        return 2 + length
    
    @staticmethod
    def estimate_workload(length: int) -> int:
        """
        Estimate the total workload for a given document length.
        
        :param length: Document length (tokens)
        :type length: int
        :return: Overall estimated workload
        :rtype: int
        """
        return Microbatch.attention_workload(length) + Microbatch.operation_workload(length)
    
class WLBOfflinePacker(OfflinePacker):
    """
    Offline implementation of workload-balanced heuristic packing algorithm from WLB-LLM (Wang et al. 2025)
    """
    def __init__(self, json_path: str, data_col: str, output_csv: str, tokenizer: AutoTokenizer, queue_bounds: List[int]):
        super().__init__(json_path, data_col, output_csv, tokenizer)
        self.packed_microbatches = []
        self.queue = OutlierQueue(queue_bounds)
        self.file_assignments = {}

    def iteration(self, docs: List[int], num_microbatches: int, max_size: int, remained_docs: List[int], iter_number: int) -> List[int]:
        """
        Run a single iteration of the WLB-LLM packing heuristic.

        :param docs: Indices of documents to attempt to pack in current iteration.
        :type docs: List[int]
        :param num_microbatches: Number of microbatches to pack per iteration.
        :type num_microbatches: int
        :param max_size: Maximum size (tokens) of a single microbatch.
        :type max_size: int
        :param remained_docs: Indices of leftover documents from previous iterations.
        :type remained_docs: List[int]
        :param iter_num: Index of current iteration.
        :type iter_number: int
        :return: Remained/leftover documents that weren't packed in current iteration.
        :rtype: List[int]
        """
        doc_lens = self.get_doc_lens(docs)

        new_docs = []

        # Push outlier documents to their respective queue, and add to new_docs otherwise
        for fname, flen in doc_lens.items():
            if not self.queue.check_outlier(fname, flen):
                new_docs.append((flen, fname))

        # Pop N outlier documents for any queue that has at least one per patch
        for q in self.queue.outlier_queues:
            if len(q) >= num_microbatches:
                for _ in range(num_microbatches):
                    new_docs.append(q.pop())

        # Sort in descending order by length
        new_docs.sort(reverse=True)

        # TODO: Can this be done faster with some method other than prepending?
        all_docs = remained_docs + new_docs
        remained_docs.clear()

        new_batches = [Microbatch(i) for i in range(num_microbatches)]

        for doc in all_docs:
            # Get microbatch w min workload and min length
            w_min = min(new_batches, key=lambda mb: mb.workload)
            l_min = min(new_batches, key=lambda mb: mb.tokens)

            # Try to add to lowest workload microbatch first
            if w_min.tokens + doc[0] <= max_size:
                w_min.push_document(doc[0], Microbatch.estimate_workload(doc[0]))
                self.file_assignments[doc[1]] = w_min.index + iter_number * num_microbatches

            # If no space, try to add to lowest length microbatch
            elif l_min.tokens + doc[0] <= max_size:
                l_min.push_document(doc[0], Microbatch.estimate_workload(doc[0]))
                self.file_assignments[doc[1]] = l_min.index + iter_number * num_microbatches

            # If neither has space, delay packing until next iteration
            else:
                remained_docs.append(doc)

        return remained_docs
    
    def pack_microbatches(self, **kwargs) -> List[List[int]]:
        """
        Pack microbatches according to WLB-LLM heuristic algorithm.

        :param kwargs: Must specify the num_microbatches (per iteration), max_size (of each microbatch), & num_iterations.
        :type kwargs: All ints
        :return: List of microbatches (each a list of document IDs).
        :rtype: List[List[int]]
        """
        num_microbatches: int = kwargs["num_microbatches"]
        max_size: int = kwargs["max_size"]
        num_iterations: int = kwargs["num_iterations"]

        if num_microbatches is None or max_size is None or num_iterations is None:
            raise KeyError("Required kwarg not given. Pack_microbatches must specify num_microbatches, max_size, and num_iterations.")
        
        dataloader = DataLoader(self.dataset, batch_size=(len(self.dataset) + num_iterations - 1) // num_iterations, 
                                shuffle=True, collate_fn=OfflinePacker.varlen_collate)
        data_iterator = iter(dataloader)

        # Loop through batches in dataloader
        remained_docs = []
        for i in range(num_iterations):
            # Iterate until end, return empty list if we hit end
            fileset = next(data_iterator, [])
            if fileset == []:
                break

            remained_docs = self.iteration(fileset, num_microbatches, max_size, remained_docs, i)

        return self.dict_to_list(self.file_assignments)
        
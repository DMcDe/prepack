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
    

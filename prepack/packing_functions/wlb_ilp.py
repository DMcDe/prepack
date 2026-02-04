from prepack.packer import OfflinePacker
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, PULP_CBC_CMD
from typing import List

def attention_workload(length: int) -> int:
    return 1 + length**2

def operation_workload(length: int) -> int:
    return 2 + length

def estimate_workload(length: int) -> int:
    return attention_workload(length) + operation_workload(length)

class WLBILPOfflinePacker(OfflinePacker):
    def pack_microbatches(self, **kwargs) -> List[List[int]]:
        """
        Pack microbatches using an ILP formulation given in WLB-LLM.
        
        :param kwargs: Must specify num_microbatches (total) & max_size (of each microbatch). Optionally specify num_threads (for ILP solver) & gap_rel (pct slack allowed for solver).
        :type kwargs: num_microbatches, max_size, num_threads are all ints. gap_rel is a float
        :return: List of microbatches (each a list of document IDs).
        :rtype: List[List[int]]
        """
        print("WARNING: If max_size is too low, PuLP will decide the problem is infeasible and will attempt to split some documents. "
              "In this case, these documents are excluded from microbatches, and results vary widely.")
        
        num_microbatches: int = kwargs["num_microbatches"]
        max_size: int = kwargs["max_size"]
        threads: int = kwargs["num_threads"] if "num_threads" in kwargs else 16
        gap: int = kwargs["gap_rel"] if "gap_rel" in kwargs else 0.03

        if num_microbatches is None or max_size is None:
            raise KeyError("Required kwarg not given. pack_microbatches must specify num_microbatches & max_size")
        
        num_docs: int = len(self.dataset)
        doc_lengths = list(self.get_doc_lens(range(num_docs)).values())

        # Create a PuLP model
        ilp = LpProblem(name="Workload_Balance", sense=LpMinimize)

        # Create a 2D binary matrix of documents and their assignments to microbatches (x_ij)
        indices = {(doc, microbatch) for doc in range(num_docs) for microbatch in range(num_microbatches)}
        xs = LpVariable.dicts("inclusion_matrix", indices=indices, lowBound=0, upBound=1, cat="Integer")

        # Constraint: each document must be assigned to a single microbatch
        for i in range(num_docs):
            ilp += (lpSum([xs[i, j] for j in range(num_microbatches)]) == 1, f"single_assignment_{i}")

        # Constraint: each microbatch must be less than the max length
        for j in range(num_microbatches):
            ilp += (lpSum([xs[i, j] * doc_lengths[i] for i in range(num_docs)]) <= max_size, f"max_mb_len_{j}")

        # Create variable to represent maximum workload of all microbatches
        # Objective is to minimize this variable
        max_workload = LpVariable("max_workload", lowBound=0, cat="Integer")
        workloads = [lpSum([xs[i, j] * estimate_workload(doc_lengths[i]) for i in range(num_docs)]) for j in range(num_microbatches)]

        # To "minimize the maximum," add constraints that max_workload must be >= all workloads
        # We set up our model as a minimization problem, so we're minimizing the max
        ilp += max_workload
        for j, wl in enumerate(workloads):
            ilp += (max_workload >= wl, f"max_workload_{j}")

        # Solve the model
        ilp.solve(PULP_CBC_CMD(threads=threads, msg=False, gapRel=gap))
        print("ILP Solution Status:", LpStatus[ilp.status])

        for v in ilp.variables():
            if v.value() == 1.0 and "inclusion_matrix" in v.name:
                # Parse variable name for microbatch and document
                tuple = v.name.split("_(")[1]
                nums = tuple.split(",_")
                d = int(nums[0])
                mb = int(nums[1][:-1])

                # Set assignment of microbatch to document
                self.file_assignments[d] = mb
            
        return self.dict_to_list(self.file_assignments)

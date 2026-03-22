from collections.abc import Sequence
import csv # TODO: Delete if unnecessary
import gzip
import json
import queue # TODO: Delete if unnecessary

class JSONLWrapper(Sequence):
    def __init__(self, json_path: str, data_col: str) -> None:
        """
        Instantiate a container to hold documents stored in a jsonl file.

        :param json_path: Path to the jsonl file holding the dataset.
        :type json_path: str
        :param data_col: Title of the column holding the data/text for the dataset.
        :type data_col: str
        """

        self.documents = []
        ofunc = gzip.open if json_path.endswith('gz') else open

        with ofunc(json_path, 'rt') as fd:
            for line in fd:
                self.documents.append(json.loads(line)[data_col])

    def __getitem__(self, index: int) -> str:
        return self.documents[index]
    
    def __len__(self) -> int:
        return len(self.documents)

# # TODO: I think this can be gotten rid of -- not actually what we need
# class JSONLRuntimeWrapper(Sequence):
#     def __init__(self, json_path: str, data_col: str, input_csv: str) -> None:
#         """
#         Instantiate a container to hold documents stored in a jsonl file.
#         Loads microbatches preemptively to parallelize with training.

#         :param json_path: Path to the jsonl file holding the dataset.
#         :type json_path: str
#         :param data_col: Title of the column holding the data/text for the dataset.
#         :type data_col: str
#         :param input_csv: Path to the CSV file holding the output of the Offline Packer.
#         :type input_csv: str
#         """

#         self.documents = queue.Queue()
#         ofunc = gzip.open if json_path.endswith('gz') else open

#         with open(self.input_csv, newline='') as input_csv:
#             reader = csv.reader(input_csv)
#             batches = list(reader)
#             self.batches = [[int(s) for s in batch] for batch in batches]

#         n = len(self.batches)
#         self.i = min(n, 256)

#         with ofunc(json_path, 'rt') as fd:
#             # TODO: Load the first XX mbs
#             for mb in range(self.i):
#                 for doc in mb:
#                     self.documents.put(json.loads())


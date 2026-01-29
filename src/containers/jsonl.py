from collections.abc import Sequence
import gzip
import json

class JSONLWrapper(Sequence):
    def __init__(self, json_path: str, data_col: str) -> None:
        """
        Instantiate a container to hold documents stored in a jsonl file.

        :param json_path: Path to the jsonl file holding the dataset.
        :type json_path: str
        :param data_col: Title of the column holding the data/text for the dataset.
        :type data_col: str
        """

        # TODO: Is this optimal? Prioritizes reducing random access latency at tradeoff of memory
        # Think I gotta change this to take the latency hit -- with large datasets this just eats up too much mem
        # Investigate orjsonl https://pypi.org/project/orjsonl/
        # Read this: https://junaid.foo/drafts/high-perf-jsonl-processing/
        # Also consider pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json

        self.documents = []
        ofunc = gzip.open if json_path.endswith('gz') else open

        with ofunc(json_path, 'rt') as fd:
            for line in fd:
                self.documents.append(json.loads(line)[data_col])

    def __getitem__(self, index: int) -> str:
        return self.documents[index]
    
    def __len__(self) -> int:
        return len(self.documents)
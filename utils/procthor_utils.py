from copy import deepcopy

import compress_pickle
import prior

from allenact.utils.system import get_logger


class Houses:
    def __init__(
        self, revision="rearrangement-2022", valid_houses_file=None,
    ):
        if valid_houses_file is None:
            self._data = prior.load_dataset("procthor-10k", revision=revision)
            self._mode = "train"
        else:
            get_logger().info(f"Using valid_houses_file {valid_houses_file}")
            self._data = {"val": compress_pickle.load(valid_houses_file)}
            self._mode = "val"

    def mode(self, mode: str):
        if mode in ["val", "valid", "validation"]:
            mode = "val"
        assert mode in [
            "train",
            "val",
            "test",
        ], f"missing {mode} (available 'train', 'val', 'test')"
        self._mode = mode

    @property
    def current_mode(self):
        return self._mode

    def __getitem__(self, pos: int):
        return deepcopy(self._data[self._mode][pos])

    def __len__(self):
        return len(self._data[self._mode])

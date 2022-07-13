import pickle

import compress_pickle
import datasets


class Houses:
    def __init__(
        self, revision="ithor-splits", valid_houses_file=None,
    ):
        if valid_houses_file is None:
            self._data = datasets.load_dataset(
                "allenai/houses", use_auth_token=True, revision=revision,
            )
            self._mode = "train"
        else:
            self._data = {"validation": compress_pickle.load(valid_houses_file)}
            self._mode = "validation"

    def mode(self, mode: str):
        if mode in ["val", "valid"]:
            mode = "validation"
        assert (
            mode in self._data
        ), f"missing {mode} (available {list(self._data.keys())})"
        self._mode = mode

    @property
    def current_mode(self):
        return self._mode

    def data(self, pos: int):
        return self[pos]

    def __getitem__(self, pos: int):
        return pickle.loads(self._data[self._mode][pos]["house"])

    def meta(self, pos: int, include_binary_house=False):
        res = {**self._data[self._mode][pos]}
        if not include_binary_house:
            res.pop("house")
        return res

    def __len__(self):
        return len(self._data[self._mode])

    def __iter__(self):
        for it, entry in enumerate(self._data[self._mode]):
            m = {**entry, "house_idx": it, "house_id": f"{self._mode}_{it}"}
            h = m.pop("house")
            yield (pickle.loads(h), m)

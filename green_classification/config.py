from glob import glob as gl
from pathlib import Path
from typing import Final

import toml
from sklearn.model_selection import train_test_split

path: Path = Path("config.toml")
if not path.exists():
    raise FileNotFoundError(
        "config.toml file not found. Please see the documentation for config.toml file."
    )

try:
    with open(path, "r") as f:
        __CONFIG: dict = toml.load(f)
        __SPLIT: Final[bool] = __CONFIG["split"]
        __DATA_PATH: Final[str] = __CONFIG["data_path"]
        __DATA_PATH_DICT: Final[dict] = __CONFIG["data"]["path"]
        __SPLIT_RATIO_DICT: Final[dict] = __CONFIG["data"]["split_ratio"]
        __TRAIN: dict = __CONFIG["train"]
        BATCH_SIZE: Final[int] = __TRAIN["batch_size"]
        EPOCHS: Final[int] = __TRAIN["num_epochs"]
        LEARNING_RATE: Final[float] = __TRAIN["learning_rate"]
    TRAIN_PATHS: list = []
    VALID_PATHS: list = []
    TEST_PATHS: list = []
    if __SPLIT:
        __dirs = gl(f"{__DATA_PATH}/*")
        for d in __dirs:
            files = gl(f"{d}/*.jpeg")
            train, test = train_test_split(
                files, test_size=__SPLIT_RATIO_DICT["test_ratio"]
            )
            train, valid = train_test_split(
                train, test_size=__SPLIT_RATIO_DICT["val_ratio"]
            )
            TRAIN_PATHS.append(train)
            VALID_PATHS.append(valid)
            TEST_PATHS.append(test)
    else:
        TRAIN_PATHS = gl(f"{__DATA_PATH_DICT['train_path']}/*.jpeg")
        VALID_PATHS = gl(f"{__DATA_PATH_DICT['val_path']}/*.jpeg")
        TEST_PATHS = gl(f"{__DATA_PATH_DICT['test_path']}/*.jpeg")
except (toml.TomlDecodeError, TypeError, KeyError) as error:
    raise TypeError(
        "config.toml file is not valid. "
        "Please see the documentation for correct config.toml file format."
    ) from error

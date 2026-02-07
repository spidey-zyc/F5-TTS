import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))
    trans_cache: dict[Path, dict[str, str]] = {}
    trans_path_cache: dict[Path, Path | None] = {}

    for line in audio_lists:
        utt_id = line.stem
        parent_dir = line.parent
        if parent_dir not in trans_cache:
            trans_files = list(parent_dir.glob("*.trans.tsv"))
            trans_map = {}
            trans_path = trans_files[0] if trans_files else None
            if trans_path:
                with open(trans_path, "r", encoding="utf-8") as f:
                    for tline in f:
                        parts = tline.rstrip("\n").split("\t")
                        if len(parts) >= 2:
                            tid = parts[0]
                            text = parts[2] if len(parts) >= 3 else parts[1]
                            trans_map[tid] = text
            trans_cache[parent_dir] = trans_map
            trans_path_cache[parent_dir] = trans_path

        trans_map = trans_cache[parent_dir]
        trans_path = trans_path_cache[parent_dir]
        # #region agent log
        global _missing_text_logged
        try:
            if "_missing_text_logged" not in globals():
                _missing_text_logged = False
            if not _missing_text_logged and (trans_path is None or utt_id not in trans_map):
                import json as _json
                from time import time as _time
                _log_path = "/hpc_stor03/sjtu_home/yichi.zhang/my_projects/.cursor/debug.log"
                with open(_log_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H4",
                                "location": "prepare_libritts.py:deal_with_audio_dir:missing_text",
                                "message": "missing transcript for wav",
                                "data": {
                                    "wav_path": str(line),
                                    "utt_id": utt_id,
                                    "parent_dir": str(parent_dir),
                                    "trans_path": str(trans_path) if trans_path else None,
                                    "trans_path_exists": trans_path.exists() if trans_path else False,
                                    "trans_map_size": len(trans_map),
                                },
                                "timestamp": int(_time() * 1000),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                _missing_text_logged = True
        except Exception:
            pass
        # #endregion agent log
        text = trans_map.get(utt_id)
        if not text:
            continue
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    # #region agent log
    try:
        import json as _json
        from time import time as _time
        _log_path = "/hpc_stor03/sjtu_home/yichi.zhang/my_projects/.cursor/debug.log"
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H3",
                        "location": "prepare_libritts.py:main:entry",
                        "message": "enter main",
                        "data": {"dataset_dir": dataset_dir, "subsets": SUB_SET, "save_dir": save_dir},
                        "timestamp": int(_time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion agent log

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    if len(result) == 0:
        # #region agent log
        try:
            import json as _json
            from time import time as _time
            _log_path = "/hpc_stor03/sjtu_home/yichi.zhang/my_projects/.cursor/debug.log"
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H5",
                            "location": "prepare_libritts.py:main:empty_result",
                            "message": "no valid samples collected; likely missing text files",
                            "data": {"dataset_dir": dataset_dir, "subsets": SUB_SET},
                            "timestamp": int(_time() * 1000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion agent log
        raise RuntimeError("No valid samples found. Check text file availability (normalized/original).")

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = "/hpc_stor03/public/shared/data/tts/LibriTTS"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours

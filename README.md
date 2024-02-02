# SVDD

This SVDD (Support Vector Data Description) system is forked from Deep-SVDD-PyTorch (https://github.com/lukasruff/Deep-SVDD-PyTorch).

## Setting up the Environment

1. 環境待更新 // 其實沒有 GPU 也可以

2. Since this training system requires data in the .npz format, we first convert the desired training CSV files (from the MOS system) into .npz format. (Change the target files "file_paths" in csv_to_npz.py.)

```
python csv_to_npz.py
```

3. make required dir

```
mkdir log
```

4. Sign up for an account at https://wandb.ai/site

## Executing Training

1. Change the Train .csv file at SVDD/src/datasets/lang_emb.py class LangEmbDataset "train_set"

2. Change the Test & Inference .csv file at SVDD/src/datasets/lang_emb.py class LangEmbDataset "test_set"

3. Change the Test & Inference result .json file at SVDD/src/npz_to_json.py

4. go to src to start training (need to log in wandb in the first time)

```
cd src
python npz_to_json.py
```

## Result

1. See the result json files and the model in SVDD/log/

2. run json_distribution.ipynb

# SVDD

This SVDD (Support Vector Data Description) system is forked from Deep-SVDD-PyTorch (https://github.com/lukasruff/Deep-SVDD-PyTorch).

## Setting up the Environment

1. @@@

```
pip
```

2. @@@

```
pip
```

3. @@@

```
pip
```

4. Since this training system requires data in the .npz format, we first convert the desired training CSV files (from the MOS system) into .npz format. (Change the target files "file_paths" in csv_to_npz.py.)

```
python csv_to_npz.py
```

5. make required dir

```
mkdir log
mkdir scores
```

## Executing Training

1. $$$

```
pip
```

2. change target test file

```
src/datasets/lang_emb.py
```

3. go to src

```
cd src
```

4. start training

```
python npz_to_json.py
```

## Result

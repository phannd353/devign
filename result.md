# Results

## v1

```bash
python train.py \
    --data data/raw/dataset.json \
    --epochs 10 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --output model/devign_token_gcn.pt

Test: loss=0.6648, accuracy=0.5837, precision=0.5358, recall=0.6522, F1=0.5883, macro-F1=0.5836, MCC=0.1789, AUC=0.6269

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.64      0.53      0.58      2229
    vulnerable       0.54      0.65      0.59      1869

      accuracy                           0.58      4098
     macro avg       0.59      0.59      0.58      4098
  weighted avg       0.59      0.58      0.58      4098

Confusion matrix:
[[1173 1056]
 [ 650 1219]]
```

## v2

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 10 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --output model/checkpoint_v2.pt

Test: loss=0.7247, accuracy=0.5886, precision=0.5370, recall=0.7111, F1=0.6119, macro-F1=0.5871, MCC=0.2006, AUC=0.6608

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.67      0.49      0.56      2229
    vulnerable       0.54      0.71      0.61      1869

      accuracy                           0.59      4098
     macro avg       0.60      0.60      0.59      4098
  weighted avg       0.61      0.59      0.58      4098

Confusion matrix:
[[1083 1146]
 [ 540 1329]]
```

## v3

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 10 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --output model/checkpoint_v3.pt

Test: loss=0.7247, accuracy=0.5915, precision=0.5414, recall=0.6816, F1=0.6035, macro-F1=0.5911, MCC=0.1990, AUC=0.6608

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.66      0.52      0.58      2229
    vulnerable       0.54      0.68      0.60      1869

      accuracy                           0.59      4098
     macro avg       0.60      0.60      0.59      4098
  weighted avg       0.61      0.59      0.59      4098

Confusion matrix:
[[1150 1079]
 [ 595 1274]]
```

## v4

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 10 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --output model/checkpoint_v4.pt


Test: loss=0.6512, accuracy=0.5979, precision=0.5412, recall=0.7774, F1=0.6381, macro-F1=0.5928, MCC=0.2355, AUC=0.6809

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.71      0.45      0.55      2229
    vulnerable       0.54      0.78      0.64      1869

      accuracy                           0.60      4098
     macro avg       0.62      0.61      0.59      4098
  weighted avg       0.63      0.60      0.59      4098

Confusion matrix:
[[ 997 1232]
 [ 416 1453]]
```

## v5

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 20 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --no-class-weights \
    --output model/checkpoint_v5.pt

py main.py --data data/raw/dataset.json --epochs 30 --output model/checkpoint_v5.pt

Test: loss=0.7199, accuracy=0.6098, precision=0.5652, recall=0.6260, F1=0.5941, macro-F1=0.6092, MCC=0.2214, AUC=0.6771

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.66      0.60      0.62      2229
    vulnerable       0.57      0.63      0.59      1869

      accuracy                           0.61      4098
     macro avg       0.61      0.61      0.61      4098
  weighted avg       0.61      0.61      0.61      4098

Confusion matrix:
[[1329  900]
 [ 699 1170]]
```

## v6

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 30 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --context-window 4 \
    --no-class-weights \
    --output model/checkpoint_v6.pt

py main.py --data data/raw/dataset.json --epochs 30 --max-tokens 512 --threshold-metric macro_f1 --context-window 4 --output model/checkpoint_v6.pt

Test: loss=0.7033, accuracy=0.6137, precision=0.5668, recall=0.6495, F1=0.6053, macro-F1=0.6135, MCC=0.2326, AUC=0.6747

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.67      0.58      0.62      2229
    vulnerable       0.57      0.65      0.61      1869

      accuracy                           0.61      4098
     macro avg       0.62      0.62      0.61      4098
  weighted avg       0.62      0.61      0.61      4098

Confusion matrix:
[[1301  928]
 [ 655 1214]]
```

## v7

```bash
python main.py \
    --data data/raw/dataset.json \
    --epochs 30 \
    --batch-size 32 \
    --hidden-dim 128 \
    --embedding-dim 128 \
    --max-tokens 512 \
    --context-window 4 \
    --no-class-weights \
    --output model/checkpoint_v7.pt

py main.py --data data/raw/dataset.json --epochs 30 --max-tokens 512 --threshold-metric macro_f1 --context-window 4 --output model/checkpoint_v7.pt

Test: loss=0.6914, accuracy=0.6232, precision=0.5819, recall=0.6180, F1=0.5994, macro-F1=0.6219, MCC=0.2448, AUC=0.6866

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.66      0.63      0.64      2229
    vulnerable       0.58      0.62      0.60      1869

      accuracy                           0.62      4098
     macro avg       0.62      0.62      0.62      4098
  weighted avg       0.63      0.62      0.62      4098

Confusion matrix:
[[1399  830]
 [ 714 1155]]
```

## TF-IDF

```bash
python3 tfidf_baseline.py \
  --data data/raw/dataset.json \
  --class-weight none

Features: 200000
Selected validation threshold: 0.41
Validation MCC: 0.2878
Test: accuracy=0.6266, precision=0.5711, recall=0.7287, f1=0.6403, macro_f1=0.6261, mcc=0.2724, auc=0.7054

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.70      0.54      0.61      2229
    vulnerable       0.57      0.73      0.64      1869

      accuracy                           0.63      4098
     macro avg       0.64      0.63      0.63      4098
  weighted avg       0.64      0.63      0.62      4098

Confusion matrix:
[[1206 1023]
 [ 507 1362]]
```

```bash
python3 tfidf_baseline.py \
  --data data/raw/dataset.json \
  --class-weight balanced

Features: 200000
Selected validation threshold: 0.45
Validation MCC: 0.2886
Test: accuracy=0.6286, precision=0.5735, recall=0.7239, f1=0.6400, macro_f1=0.6282, mcc=0.2747, auc=0.7055

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.70      0.55      0.62      2229
    vulnerable       0.57      0.72      0.64      1869

      accuracy                           0.63      4098
     macro avg       0.64      0.64      0.63      4098
  weighted avg       0.64      0.63      0.63      4098

Confusion matrix:
[[1223 1006]
 [ 516 1353]]
```

## RGCN attention AST dataflow

```bash
python3 main.py   \
    --data data/raw/dataset.json   \
    --epochs 30   \
    --batch-size 32   \
    --embedding-dim 128   \
    --hidden-dim 128   \
    --max-tokens 512   \
    --context-window 2  \
    --ast-edges   \
    --data-flow-edges   \
    --output model/rgcn_attention_ast_dataflow.pt

Selected decision threshold: 0.28 (mcc=0.2426)

Final test result:
Test: loss=0.6563, accuracy=0.5593, precision=0.5096, recall=0.8941, F1=0.6492, macro-F1=0.5283, MCC=0.2151, AUC=0.6644

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.76      0.28      0.41      2229
    vulnerable       0.51      0.89      0.65      1869

      accuracy                           0.56      4098
     macro avg       0.63      0.59      0.53      4098
  weighted avg       0.64      0.56      0.52      4098

Confusion matrix:
[[ 621 1608]
 [ 198 1671]]
```

## RGCN fixed attention

```bash
python3 main.py   \
    --data data/raw/dataset.json   \
    --epochs 30   \
    --batch-size 32   \
    --embedding-dim 128   \
    --hidden-dim 128   \
    --max-tokens 512   \
    --context-window 2  \
    --ast-edges   \
    --data-flow-edges   \
    --output model/rgcn_attention_ast_dataflow.pt

Selected decision threshold: 0.51 (macro_f1=0.6252)

Final test result:
Test: loss=0.6699, accuracy=0.6232, precision=0.5909, recall=0.5650, F1=0.5777, macro-F1=0.6188, MCC=0.2381, AUC=0.6840

Classification report:
                precision    recall  f1-score   support

non-vulnerable       0.65      0.67      0.66      2229
    vulnerable       0.59      0.57      0.58      1869

      accuracy                           0.62      4098
     macro avg       0.62      0.62      0.62      4098
  weighted avg       0.62      0.62      0.62      4098

Confusion matrix:
[[1498  731]
 [ 813 1056]]
```

## Federated learning

```bash
flwr run . --stream --run-config "local_epochs=10" --run-config "rounds=5"

Final results:
INFO :
INFO :      	Global Arrays:
INFO :      		ArrayRecord (11.688 MB)
INFO :
INFO :      	Aggregated ClientApp-side Train Metrics:
INFO :      	{ 1: {'train-loss': '5.9710e-01'},
INFO :      	 2: {'train-loss': '3.7426e-01'},
INFO :      	 3: {'train-loss': '1.7633e-01'},
INFO :      	 4: {'train-loss': '9.5333e-02'},
INFO :      	 5: {'train-loss': '8.0640e-02'}}
INFO :
INFO :      	Aggregated ClientApp-side Evaluate Metrics:
INFO :      	{ 1: { 'accuracy': '5.5147e-01',
INFO :      	      'auc': '6.1696e-01',
INFO :      	      'loss': '6.8960e-01',
INFO :      	      'mcc': '1.8787e-01'},
INFO :      	 2: { 'accuracy': '5.9559e-01',
INFO :      	      'auc': '6.8838e-01',
INFO :      	      'loss': '7.0669e-01',
INFO :      	      'mcc': '2.2747e-01'},
INFO :      	 3: { 'accuracy': '6.3971e-01',
INFO :      	      'auc': '6.9075e-01',
INFO :      	      'loss': '9.6606e-01',
INFO :      	      'mcc': '2.6065e-01'},
INFO :      	 4: { 'accuracy': '6.3235e-01',
INFO :      	      'auc': '6.9599e-01',
INFO :      	      'loss': '1.3130e+00',
INFO :      	      'mcc': '2.3589e-01'},
INFO :      	 5: { 'accuracy': '6.3235e-01',
INFO :      	      'auc': '6.8722e-01',
INFO :      	      'loss': '1.4793e+00',
INFO :      	      'mcc': '2.2944e-01'}}
INFO :
INFO :      	ServerApp-side Evaluate Metrics:
INFO :      	{ 0: { 'accuracy': '5.4392e-01',
INFO :      	      'auc': '4.7855e-01',
INFO :      	      'loss': '7.0167e-01',
INFO :      	      'mcc': '0.0000e+00'},
INFO :      	 1: { 'accuracy': '5.7321e-01',
INFO :      	      'auc': '5.9364e-01',
INFO :      	      'loss': '6.8616e-01',
INFO :      	      'mcc': '1.2409e-01'},
INFO :      	 2: { 'accuracy': '5.6552e-01',
INFO :      	      'auc': '6.0228e-01',
INFO :      	      'loss': '8.6201e-01',
INFO :      	      'mcc': '1.4356e-01'},
INFO :      	 3: { 'accuracy': '5.5893e-01',
INFO :      	      'auc': '5.9947e-01',
INFO :      	      'loss': '1.4015e+00',
INFO :      	      'mcc': '1.3944e-01'},
INFO :      	 4: { 'accuracy': '5.7613e-01',
INFO :      	      'auc': '6.0071e-01',
INFO :      	      'loss': '1.9668e+00',
INFO :      	      'mcc': '1.6775e-01'},
INFO :      	 5: { 'accuracy': '5.7028e-01',
INFO :      	      'auc': '6.0313e-01',
INFO :      	      'loss': '2.2502e+00',
INFO :      	      'mcc': '1.5770e-01'}}
```

split-niid:
	python3 split_federated_dataset.py \
      --input data/raw/dataset.json \
      --output-dir data/federated \
      --mode non-iid \
      --seed 42

split-iid:
	python3 split_federated_dataset.py \
      --input data/raw/dataset.json \
      --output-dir data/federated \
      --mode iid \
      --seed 42

build-vocab:
	# max 53752 tokens
	python3 export_vocabulary.py \
      --data data/federated/iid/server/test.json \
      --output data/federated/vocabulary.json \
      --max-vocab-size 20000

start-server:
	python3 fl_server.py \
      --vocabulary data/federated/vocabulary.json \
      --server-address 0.0.0.0:8080 \
      --rounds 20 \
      --min-clients 2 \
      --output federated_checkpoint.pt

start-client:
	python3 fl_client.py \
	  --server-address 127.0.0.1:8080 \
	  --data client_1/data.json \
	  --vocabulary data/federated/vocabulary.json \
	  --local-epochs 1 \
	  --batch-size 32 \
	  --max-tokens 512 \
	  --context-window 2 \
	  --ast-edges \
	  --data-flow-edges

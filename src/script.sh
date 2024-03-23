#!/bin/bash

python train_pos.py --config_path ./configs/postagging/config-lstm-en.yaml --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-en.yaml --embedding_path ./outputs/ud_en/ud_en_embedding --language en --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-en.yaml --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-en.yaml --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en --use_wandb


python train_pos.py --config_path ./configs/postagging/config-lstm-sk.yaml --embedding_path ./outputs/combined_sk/combined_sk_embedding --language sk --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-sk.yaml --embedding_path ./outputs/slovaksum_sk/slovaksum_sk_embedding --language sk --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-sk.yaml --embedding_path ./outputs/squadsk_sk/squad_sk_embedding --language sk --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-sk.yaml --embedding_path ./outputs/ud_sk/ud_sk_embedding --language sk --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-sk.yaml --embedding_path ./outputs/wikipedia_sk/wikipedia_sk_embedding --language sk --use_wandb


python train_pos.py --config_path ./configs/postagging/config-lstm-cs.yaml --embedding_path ./outputs/combined_cs/combined_cs_embedding --language cs --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-cs.yaml --embedding_path ./outputs/ud_cs/ud_cs_embedding --language cs --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-cs.yaml --embedding_path ./outputs/sumeczech_cs/sumeczech_cs_embedding --language cs --use_wandb
python train_pos.py --config_path ./configs/postagging/config-lstm-cs.yaml --embedding_path ./outputs/wikipedia_cs/wikipedia_cs_embedding --language cs --use_wandb

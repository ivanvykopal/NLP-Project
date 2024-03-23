#!/bin/bash

#sk-demagog
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/slovaksum_sk/slovaksum_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/wikipedia_sk/wikipedia_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/ud_sk/ud_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/combined_sk/combined_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/squadsk_sk/squad_sk_embedding --language sk

#sk-combined
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_sk --embedding_path ./outputs/slovaksum_sk/slovaksum_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_sk --embedding_path ./outputs/wikipedia_sk/wikipedia_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_sk --embedding_path ./outputs/ud_sk/ud_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_sk --embedding_path ./outputs/combined_sk/combined_sk_embedding --language sk
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_sk --embedding_path ./outputs/squadsk_sk/squad_sk_embedding --language sk

#cs-csfever
python train_downstream.py --config_path ./configs/config.yaml --dataset csfever --embedding_path ./outputs/sumeczech_cs/sumeczech_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset csfever --embedding_path ./outputs/wikipedia_cs/wikipedia_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset csfever --embedding_path ./outputs/ud_cs/ud_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset csfever --embedding_path ./outputs/combined_cs/combined_cs_embedding --language cs


#cs-ctkfacts
python train_downstream.py --config_path ./configs/config.yaml --dataset ctkfacts --embedding_path ./outputs/sumeczech_cs/sumeczech_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset ctkfacts --embedding_path ./outputs/wikipedia_cs/wikipedia_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset ctkfacts --embedding_path ./outputs/ud_cs/ud_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset ctkfacts --embedding_path ./outputs/combined_cs/combined_cs_embedding --language cs

#cs-demagog
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/sumeczech_cs/sumeczech_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/wikipedia_cs/wikipedia_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/ud_cs/ud_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset demagog --embedding_path ./outputs/combined_cs/combined_cs_embedding --language cs

#cs-combined
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_cs --embedding_path ./outputs/sumeczech_cs/sumeczech_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_cs --embedding_path ./outputs/wikipedia_cs/wikipedia_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_cs --embedding_path ./outputs/ud_cs/ud_cs_embedding --language cs
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_cs --embedding_path ./outputs/combined_cs/combined_cs_embedding --language cs

#en-fever
python train_downstream.py --config_path ./configs/config.yaml --dataset fever --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset fever --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset fever --embedding_path ./outputs/ud_en/ud_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset fever --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en

#en-liar
python train_downstream.py --config_path ./configs/config.yaml --dataset liar --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset liar --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset liar --embedding_path ./outputs/ud_en/ud_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset liar --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en

#en-multifc
python train_downstream.py --config_path ./configs/config.yaml --dataset multifc --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset multifc --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset multifc --embedding_path ./outputs/ud_en/ud_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset multifc --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en

#en-x-fact
python train_downstream.py --config_path ./configs/config.yaml --dataset xfact --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset xfact --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset xfact --embedding_path ./outputs/ud_en/ud_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset xfact --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en

#en-combined
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_en --embedding_path ./outputs/ccnews_en/ccnews_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_en --embedding_path ./outputs/fasttext/wikipedia_en_embedding/wikipedia_en_embedding.ft --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_en --embedding_path ./outputs/ud_en/ud_en_embedding --language en
python train_downstream.py --config_path ./configs/config.yaml --dataset combined_claim_en --embedding_path ./outputs/fasttext/combined_en_embedding/combined_en_embedding.ft --language en

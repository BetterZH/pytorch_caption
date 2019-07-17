7500
15000
22500


CUDA_VISIBLE_DEVICES=0 python eval_aic.py --start 0 --num 7500 \
--start_from_best /home/amds/caption/caption_aic_best/aic_Bleu_4 \
--id aic_Bleu_4

CUDA_VISIBLE_DEVICES=1 python eval_aic.py --start 7500 --num 7500 \
--start_from_best /home/amds/caption/caption_aic_best/aic_Bleu_4 \
--id aic_Bleu_4

CUDA_VISIBLE_DEVICES=2 python eval_aic.py --start 15000 --num 7500 \
--start_from_best /home/amds/caption/caption_aic_best/aic_Bleu_4 \
--id aic_Bleu_4

CUDA_VISIBLE_DEVICES=3 python eval_aic.py --start 22500 --num 7500 \
--start_from_best /home/amds/caption/caption_aic_best/aic_Bleu_4 \
--id aic_Bleu_4
pip install --user -r requirements.txt 

python preprocess.py --source_font fonts/SentyCreamPuff.ttf --target_font fonts/SentyCHALKoriginal.ttf --char_list charsets/top_500_simplified.txt --save_dir path_to_save_bitmap

python rewrite.py --mode=train --model=small --source_font=path_to_save_bitmap/SentyCHALKoriginal.npy --target_font=path_to_save_bitmap/SentyCreamPuff.npy --iter=3000 --num_examples=2100 --num_validations=100 --tv=0.0001 --alpha=0.2 --keep_prob=0.9 --num_ckpt=10 --ckpt_dir=path_to_save_checkpoints --summary_dir=path_to_save_summaries --frame_dir=path_to_save_frames
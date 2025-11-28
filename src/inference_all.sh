# LJSpeech_SNR_10dB_split_5s.npz
# LJ_random.npz
# LJ_random_split_5s.npz
# LJSpeech.npz
# LJSpeech_split_5s.npz
# distortion_SNR_10dB.npz
# matbn_enhanced_split_5s

python npz_to_json.py --load_model ../log_LJ/model.tar --n_epochs 0 --pretrain false  --xp_path ../log_LJ  --net_name lang_emb_LeNet --dataset_name lang_emb


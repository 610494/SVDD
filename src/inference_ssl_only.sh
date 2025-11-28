# matbn_enhanced_split_5s_ssl_only.npz

# LJSpeech_split_5s_ssl.npz
# LJ_random_split_5s_ssl.npz
# LJSpeech_ssl.npz
# matbn_enhanced_ssl_only.npz

# LJSpeech_SNR_10dB_split_5s_ssl_only.npz
# distortion_SNR_10dB_ssl_only.npz
# LJ_random_ssl_only.npz

# python npz_to_json.py --load_model ../log_LJ_ssl/model.tar --n_epochs 0 --pretrain false  --xp_path ../log_LJ_ssl  --net_name lang_emb_LeNet_ssl --dataset_name lang_emb_ssl
import pandas as pd
import numpy as np

file_paths = [
    '/mnt/md1/user_wago/data/aishell3/aishell3_train_len_free_select_3.5.csv'
]

for file_path in file_paths:
    df = pd.read_csv(file_path)

    ids = df['filename'].values
    features = df.drop(['filename', 'len_in_sec'], axis=1).values
    # features_remove_SIG_MOS = df.drop(['filename', 'len_in_sec', 'MOS_COL','MOS_DISC','MOS_LOUD','MOS_NOISE','MOS_REVERB','MOS_SIG','MOS_OVRL'], axis=1).values
    # features_dns_ssl = df.drop(['filename', 'len_in_sec', 'MOS_COL','MOS_DISC','MOS_LOUD','MOS_NOISE','MOS_REVERB','MOS_SIG','MOS_OVRL','mos_pred','noi_pred','dis_pred','col_pred','loud_pred'], axis=1).values
    features_len_free = df.drop(['filename', 'len_in_sec', 'MOS_COL','MOS_DISC','MOS_LOUD','MOS_NOISE','MOS_REVERB','MOS_SIG','MOS_OVRL', 'OVRL_raw', 'SIG_raw', 'BAK_raw', 'OVRL', 'SIG', 'BAK', 'P808_MOS', 'col_pred','loud_pred'], axis=1).values
    # features_nisqa_ssl = df.drop(['filename','len_in_sec','mos_pred','dis_pred','col_pred','loud_pred','MOS_COL','MOS_LOUD','MOS_REVERB','MOS_SIG','MOS_OVRL'], axis=1).values
    # filename,len_in_sec,OVRL_raw,SIG_raw,BAK_raw,OVRL,SIG,BAK,P808_MOS,mos_pred,noi_pred,dis_pred,col_pred,loud_pred,MOS_SSL,MOS_COL,MOS_DISC,MOS_LOUD,MOS_NOISE,MOS_REVERB,MOS_SIG,MOS_OVRL
    # features_only_dns_mos = df[['OVRL_raw','SIG_raw','BAK_raw','OVRL','SIG','BAK','P808_MOS']]

    npz_file_path = file_path.replace('.csv', '.npz')
    # npz_file_path_dns_ssl = file_path.replace('.csv', '_dns_ssl.npz') 
    # npz_file_path_nisqa_ssl = file_path.replace('.csv', '_nisqa_ssl.npz') 
    npz_file_path_len_free = file_path.replace('.csv', '_len_free.npz') 
    # npz_file_path_remove_SIG_MOS = file_path.replace('.csv', '_remove_SIG_MOS.npz')
    # npz_file_path_only_dns_mos = file_path.replace('.csv', '_only_dns_mos.npz')

    np.savez(npz_file_path, ids=ids, features=features)
    # np.savez(npz_file_path_dns_ssl, ids=ids, features=features_dns_ssl)
    # np.savez(npz_file_path_nisqa_ssl, ids=ids, features=features_nisqa_ssl)
    np.savez(npz_file_path_len_free, ids=ids, features=features_len_free)
    # np.savez(npz_file_path_remove_SIG_MOS, ids=ids, features=features_remove_SIG_MOS)
    # np.savez(npz_file_path_only_dns_mos, ids=ids, features=features_only_dns_mos)
    # print(f"Saved {npz_file_path}.")

from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .lang_emb import LangEmbDataset
from .lang_emb_no_sig import LangEmbNoSigDataset
from .lang_emb_ssl import LangEmbSllDataset
from .lang_emb_dns_ssl import LangEmbDnsSllDataset
from .lang_emb_nisqa_ssl import LangEmbNisqaSllDataset
from .lang_emb_len_free import LangEmbLenFreeDataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'lang_emb', 'lang_emb_no_sig', 'lang_emb_ssl', 'lang_emb_dns_ssl', 'lang_emb_nisqa_ssl', 'lang_emb_len_free')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'lang_emb':
        # dataset = LangEmbDataset(normal_class=normal_class)
        dataset = LangEmbDataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'lang_emb_no_sig':
        # dataset = LangEmbDataset(normal_class=normal_class)
        dataset = LangEmbNoSigDataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'lang_emb_ssl':
        # dataset = LangEmbDataset(normal_class=normal_class)
        dataset = LangEmbSllDataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'lang_emb_dns_ssl':
        dataset = LangEmbDnsSllDataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'lang_emb_nisqa_ssl':
        dataset = LangEmbNisqaSllDataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'lang_emb_len_free':
        dataset = LangEmbLenFreeDataset(root=data_path, normal_class=normal_class)
    
    return dataset

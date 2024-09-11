from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .lang_emb_LeNet import LangEmbLeNet, LangEmbLeNetAutoencoder
from .lang_emb_LeNet_no_SIG import LangEmbLeNetNoSIG, LangEmbLeNetNoSIGAutoencoder
from .lang_emb_LeNet_ssl import LangEmbLeNetSll, LangEmbLeNetSllAutoencoder
from .lang_emb_LeNet_dns_ssl import LangEmbLeNetDnsSll, LangEmbLeNetDnsSllAutoencoder
from .lang_emb_LeNet_nisqa_ssl import LangEmbLeNetNisqaSll, LangEmbLeNetNisqaSllAutoencoder
from .lang_emb_LeNet_len_free import LangEmbLeNetLenFree, LangEmbLeNetLenFreeAutoencoder





def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'lang_emb_LeNet', 'lang_emb_LeNet_no_SIG', 'lang_emb_LeNet_ssl', 'lang_emb_LeNet_dns_ssl', 'lang_emb_LeNet_nisqa_ssl', 'lang_emb_LeNet_len_free')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()
        
    if net_name == 'lang_emb_LeNet':
        net = LangEmbLeNet()

    if net_name == 'lang_emb_LeNet_no_SIG':
        net = LangEmbLeNetNoSIG()
    
    if net_name == 'lang_emb_LeNet_ssl':
        net = LangEmbLeNetSll()
    
    if net_name == 'lang_emb_LeNet_dns_ssl':
        net = LangEmbLeNetDnsSll()
    
    if net_name == 'lang_emb_LeNet_nisqa_ssl':
        net = LangEmbLeNetNisqaSll()
        
    if net_name == 'lang_emb_LeNet_len_free':
        net = LangEmbLeNetLenFree()
    
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'lang_emb_LeNet', 'lang_emb_LeNet_no_SIG', 'lang_emb_LeNet_ssl', 'lang_emb_LeNet_dns_ssl', 'lang_emb_LeNet_nisqa_ssl', 'lang_emb_LeNet_len_free')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'lang_emb_LeNet':
        ae_net = LangEmbLeNetAutoencoder()
        
    if net_name == 'lang_emb_LeNet_no_SIG':
        ae_net = LangEmbLeNetNoSIGAutoencoder()
    
    if net_name == 'lang_emb_LeNet_ssl':
        ae_net = LangEmbLeNetSllAutoencoder()
    
    if net_name == 'lang_emb_LeNet_dns_ssl':
        ae_net = LangEmbLeNetDnsSllAutoencoder()
        
    if net_name == 'lang_emb_LeNet_nisqa_ssl':
        ae_net = LangEmbLeNetNisqaSllAutoencoder()
        
    if net_name == 'lang_emb_LeNet_len_free':
        ae_net = LangEmbLeNetLenFreeAutoencoder()
    
    return ae_net

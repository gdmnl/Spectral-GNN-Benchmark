from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plt_tsne(embeddings, labels, save_path=None, title='Title'):
    print('Drawing t-SNE plot ...')
    print("Embedding size:", embeddings.shape)
    tsne = TSNE(n_components = 2, perplexity = 30.0, early_exaggeration = 12,
               n_iter = 1000, learning_rate = 368)

    emb = tsne.fit_transform(embeddings.cpu().numpy())
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels.cpu().numpy(), marker='o', cmap='Set2')
    # plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


def test_tsne(trn):
    input, label = trn._fetch_input()
    def plt_tsne_hook(module, input, output):
        plt_tsne(input[0].detach(), label, save_path=trn.logpath.joinpath("tsne.pdf"))

    trn.model.eval()
    trn.model.out_mlp.register_forward_hook(plt_tsne_hook)
    trn.model(*input)

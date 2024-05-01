# from openTSNE import TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plt(embeddings, labels, save_path=None, title='Title'):
    print('Drawing t-SNE plot ...')
    print("Embedding size:", embeddings.shape)
    # tsne = TSNE(n_components=2, perplexity=30, metric="euclidean", n_jobs=40, random_state=42, verbose=False)
    tsne = TSNE(n_components = 2, perplexity = 30.0, early_exaggeration = 12, 
               n_iter = 1000, learning_rate = 368, verbose = 1)

    embeddings = embeddings.cpu().numpy()
    c = labels.cpu().numpy()

    # emb = tsne.fit(embeddings)  # Training
    emb = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
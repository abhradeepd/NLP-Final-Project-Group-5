from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.graph_objs as go


def plot_tsne_visualization(data, n_iter=1000, n_components=3, subset_size=5000, perplexity=30):
    """
    Generate a t-SNE visualization with a configurable perplexity parameter.

    Args:
        data: DataFrame containing features and labels.
        n_iter: Number of iterations for t-SNE.
        n_components: Number of dimensions for t-SNE (default is 3 for 3D plotting).
        subset_size: Number of samples to use for the visualization.
        perplexity: Perplexity parameter for t-SNE.

    Returns:
        Plotly 3D scatter plot figure.
    """
    print(f"Running t-SNE with perplexity={perplexity}...")
    subset_size = min(subset_size, len(data))
    dfp_subsampled = data[:subset_size]

    # Scaling features
    X = MinMaxScaler().fit_transform(dfp_subsampled[['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
                                                     'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
                                                     'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio',
                                                     'fuzz_partial_ratio', 'longest_substr_ratio']])
    y = dfp_subsampled['is_duplicate'].values
    hover_text = dfp_subsampled.apply(lambda row: f"Q1: {row['question1']}<br>Q2: {row['question2']}", axis=1)

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        init='random',
        random_state=101,
        method='barnes_hut',
        n_iter=n_iter,
        perplexity=perplexity,
        verbose=2,
        angle=0.5
    ).fit_transform(X)

    if n_components == 3:
        # Separate data by label
        tsne_non_duplicate = tsne[y == 0]
        tsne_duplicate = tsne[y == 1]

        # Create traces for each class
        trace_non_duplicate = go.Scatter3d(
            x=tsne_non_duplicate[:, 0],
            y=tsne_non_duplicate[:, 1],
            z=tsne_non_duplicate[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='blue',  # Blue for Non-Duplicate
                opacity=0.7
            ),
            name='Non-Duplicate',
            text=hover_text[y == 0],  # Hover text for Non-Duplicates
            hoverinfo='text'
        )

        trace_duplicate = go.Scatter3d(
            x=tsne_duplicate[:, 0],
            y=tsne_duplicate[:, 1],
            z=tsne_duplicate[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='red',  # Red for Duplicate
                opacity=0.7
            ),
            name='Duplicate',
            text=hover_text[y == 1],  # Hover text for Duplicates
            hoverinfo='text'
        )

        # Combine traces
        fig = go.Figure(data=[trace_non_duplicate, trace_duplicate])
        fig.update_layout(
            height=800,
            width=800,
            title=f'3D TSNE Visualization (Perplexity={perplexity})',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z')
            )
        )
        return fig
    else:
        raise ValueError("Only 3D plotting is supported with this configuration.")

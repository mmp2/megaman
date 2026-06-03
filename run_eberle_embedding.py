import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        '01062026_sphaera_textparts',
                        'text_parts_with_metadata')

GT_PATH   = os.path.join(DATA_DIR, 'ground_truth_matrix_of_text_parts.npy')
SIM_PATH  = os.path.join(DATA_DIR, 'similarity_matrix_of_text_parts.npy')
META_PATH = os.path.join(DATA_DIR, 'text_parts_with_metadata.csv')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'eberle_results')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_COMPONENTS   = 10            # eigenvectors to compute (Prof. Meila's advice)
LAPLACIAN_TYPE = 'geometric'   # Coifman-Lafon: best for manifold structure
EIGEN_SOLVER   = 'arpack'      # robust for sparse/large matrices
RANDOM_STATE   = 42

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    gt   = np.load(GT_PATH)
    sim  = np.load(SIM_PATH)
    meta = pd.read_csv(META_PATH)

    # The CSV has 1942 rows (1 extra header row artifact?) — align to matrix
    if len(meta) == gt.shape[0] + 1:
        meta = meta.iloc[1:].reset_index(drop=True)

    print(f"  Ground truth matrix : {gt.shape},  range [{gt.min():.3f}, {gt.max():.3f}]")
    print(f"  TF-IDF sim matrix   : {sim.shape}, range [{sim.min():.3f}, {sim.max():.3f}]")
    print(f"  Metadata rows       : {len(meta)}, columns: {list(meta.columns)}")
    return gt, sim, meta


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Prepare affinity matrices
# ─────────────────────────────────────────────────────────────────────────────

def prepare_affinity(matrix: np.ndarray, label: str) -> sp.csr_matrix:
    """
    Clip negatives, symmetrize, and convert to sparse CSR.
    The TF-IDF matrix has small negative values (down to -0.71) which
    are not valid affinities — we zero them out.
    """
    A = matrix.copy()

    # Clip negatives to 0 (cosine sim can go slightly negative for dissimilar docs)
    neg_count = (A < 0).sum()
    if neg_count > 0:
        print(f"  [{label}] Clipping {neg_count} negative values to 0")
        A = np.clip(A, 0, None)

    # Ensure symmetry
    A = 0.5 * (A + A.T)

    # Convert to sparse (keep only non-zero entries)
    A_sparse = sp.csr_matrix(A)
    print(f"  [{label}] Sparse affinity: nnz={A_sparse.nnz} "
          f"({100*A_sparse.nnz / A.size:.1f}% non-zero)")
    return A_sparse


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run embedding
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding(affinity_matrix: sp.csr_matrix, label: str):
    """
    Inject precomputed affinity into a Geometry object and run
    Spectral Embedding (Laplacian Eigenmaps).

    Returns
    -------
    embedding    : ndarray (n_docs, N_COMPONENTS)
    eigenvalues  : ndarray (N_COMPONENTS,)
    eigenvectors : ndarray (n_docs, N_COMPONENTS)
    laplacian    : sparse matrix (n_docs, n_docs)   ← for distortion computation
    """
    print(f"\n{'='*60}")
    print(f"  Embedding: {label}")
    print(f"{'='*60}")

    # Build Geometry and inject the precomputed affinity (bypasses
    # adjacency/distance computation entirely)
    geom = Geometry(laplacian_method=LAPLACIAN_TYPE)
    geom.set_affinity_matrix(affinity_matrix)

    # Compute Laplacian with full output (needed for distortion + lobpcg solvers)
    print("  Computing Laplacian ...")
    geom.compute_laplacian_matrix(copy=False, return_lapsym=True)
    print(f"  Laplacian type: {LAPLACIAN_TYPE}")

    # SpectralEmbedding with precomputed affinity
    print(f"  Eigen-decomposition (n_components={N_COMPONENTS}, solver={EIGEN_SOLVER}) ...")
    se = SpectralEmbedding(
        n_components   = N_COMPONENTS,
        geom           = geom,
        eigen_solver   = EIGEN_SOLVER,
        random_state   = RANDOM_STATE,
        drop_first     = True,   # drop trivial constant eigenvector
        diffusion_maps = False,
    )
    se.fit(affinity_matrix, input_type='affinity')

    embedding    = se.embedding_        # (n_docs, N_COMPONENTS)
    eigenvalues  = se.eigenvalues_      # (N_COMPONENTS,)
    eigenvectors = se.eigenvectors_     # (n_docs, N_COMPONENTS)
    laplacian    = se.laplacian_matrix_

    print(f"  Embedding shape : {embedding.shape}")
    print(f"  Eigenvalues     : {np.round(eigenvalues, 5)}")

    # ── Save outputs ─────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = os.path.join(OUT_DIR, label)

    np.save(f"{prefix}_embedding.npy",    embedding)
    np.save(f"{prefix}_eigenvalues.npy",  eigenvalues)
    np.save(f"{prefix}_eigenvectors.npy", eigenvectors)
    sp.save_npz(f"{prefix}_laplacian.npz", sp.csr_matrix(laplacian))

    print(f"\n  Saved:")
    print(f"    {label}_embedding.npy     shape={embedding.shape}")
    print(f"    {label}_eigenvalues.npy   shape={eigenvalues.shape}")
    print(f"    {label}_eigenvectors.npy  shape={eigenvectors.shape}")
    print(f"    {label}_laplacian.npz     sparse Laplacian (for distortion)")

    return embedding, eigenvalues, eigenvectors, laplacian


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def _parse_year(year_val):
    """Parse year column — may be a string range like '1550-1600' or a plain int."""
    try:
        s = str(year_val).strip()
        if '-' in s and not s.startswith('-'):
            return float(s.split('-')[0])
        return float(s)
    except Exception:
        return np.nan


def plot_embedding_grid(embedding: np.ndarray, label: str, meta: pd.DataFrame):
    """
    3-panel figure:
      1. Dims 1 & 2, colored by document index
      2. Dims 1 & 2, colored by year
      3. Dims 2 & 3, colored by language
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Spectral Embedding — {label}  ({embedding.shape[0]} documents)',
                 fontsize=13, fontweight='bold')

    x1, x2 = embedding[:, 0], embedding[:, 1]
    x3 = embedding[:, 2] if embedding.shape[1] > 2 else embedding[:, 0]

    # Panel 1: colored by document index
    ax = axes[0]
    sc = ax.scatter(x1, x2, s=8, alpha=0.6, c=np.arange(len(x1)), cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Doc index')
    ax.set_xlabel('Eigenvec 1'); ax.set_ylabel('Eigenvec 2')
    ax.set_title('Dims 1 & 2 (by index)')

    # Panel 2: colored by year
    ax = axes[1]
    years = meta['years'].apply(_parse_year).values
    valid = ~np.isnan(years)
    sc = ax.scatter(x1[~valid], x2[~valid], s=6, alpha=0.3, c='lightgray')
    sc = ax.scatter(x1[valid],  x2[valid],  s=8, alpha=0.7,
                    c=years[valid], cmap='plasma')
    plt.colorbar(sc, ax=ax, label='Year')
    ax.set_xlabel('Eigenvec 1'); ax.set_ylabel('Eigenvec 2')
    ax.set_title('Dims 1 & 2 (by year)')

    # Panel 3: colored by language
    ax = axes[2]
    langs = meta['languages'].fillna('Unknown').values
    unique_langs = sorted(set(langs))
    lang_cmap = cm.get_cmap('tab20', len(unique_langs))
    lang_to_idx = {l: i for i, l in enumerate(unique_langs)}
    colors = [lang_to_idx[l] for l in langs]
    sc = ax.scatter(x1, x3, s=8, alpha=0.6, c=colors, cmap='tab20',
                    vmin=0, vmax=len(unique_langs)-1)
    # Legend (top languages only)
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=lang_cmap(lang_to_idx[l]), markersize=6,
                          label=l)
               for l in unique_langs[:12]]
    ax.legend(handles=handles, fontsize=6, loc='upper right', framealpha=0.7)
    ax.set_xlabel('Eigenvec 1'); ax.set_ylabel('Eigenvec 3')
    ax.set_title('Dims 1 & 3 (by language)')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{label}_embedding_plots.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def plot_eigenvalue_spectra(ev_dict: dict):
    """Compare eigenvalue spectra across both similarity matrices."""
    fig, ax = plt.subplots(figsize=(8, 4))
    markers = ['o-', 's--']
    for (label, evals), mk in zip(ev_dict.items(), markers):
        ax.plot(np.arange(1, len(evals)+1), evals, mk, label=label,
                markersize=6, linewidth=1.8)
    ax.set_xlabel('Eigenvector index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue spectra — look for the "gap" to choose embedding dim')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(OUT_DIR, 'eigenvalue_spectra.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Eigenvalue plot saved: {out_path}")


def plot_gt_vs_tfidf(emb_gt: np.ndarray, emb_tfidf: np.ndarray):
    """Side-by-side comparison of GT vs TF-IDF embeddings (dims 1 & 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('GT vs TF-IDF: Dims 1 & 2', fontsize=13, fontweight='bold')

    for ax, emb, title in zip(axes,
                               [emb_gt, emb_tfidf],
                               ['Ground Truth Similarity', 'TF-IDF Cosine Similarity']):
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=7, alpha=0.6,
                        c=np.arange(len(emb)), cmap='viridis')
        plt.colorbar(sc, ax=ax, label='Doc index')
        ax.set_xlabel('Eigenvec 1'); ax.set_ylabel('Eigenvec 2')
        ax.set_title(title)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'gt_vs_tfidf_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    gt_raw, sim_raw, meta = load_data()

    # ── Prepare affinities ────────────────────────────────────────────────
    print("\nPreparing affinity matrices ...")
    A_gt   = prepare_affinity(gt_raw,  'gt')
    A_tfidf = prepare_affinity(sim_raw, 'tfidf')

    # ── Run embeddings ────────────────────────────────────────────────────
    emb_gt,   evals_gt,   _, lap_gt   = run_embedding(A_gt,    'gt')
    emb_tfidf, evals_tfidf, _, lap_tfidf = run_embedding(A_tfidf, 'tfidf')

    # ── Visualize ─────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_embedding_grid(emb_gt,    'gt',    meta)
    plot_embedding_grid(emb_tfidf, 'tfidf', meta)
    plot_gt_vs_tfidf(emb_gt, emb_tfidf)
    plot_eigenvalue_spectra({'gt': evals_gt, 'tfidf': evals_tfidf})

    print(f"\n{'='*60}")
    print(f"  Done!  All results saved to: {OUT_DIR}/")
    print(f"{'='*60}")
    print("""
Outputs:
  gt_embedding.npy          ← 1941 × 10  (eigenvectors)
  gt_eigenvalues.npy        ← (10,)
  gt_eigenvectors.npy       ← 1941 × 10
  gt_laplacian.npz          ← sparse Laplacian for distortion
  tfidf_* (same set)

  gt_embedding_plots.png    ← 3-panel: by index / year / language
  tfidf_embedding_plots.png
  gt_vs_tfidf_comparison.png
  eigenvalue_spectra.png    ← look for spectral gap

Discussion points for meeting:
  1. Do GT and TF-IDF embeddings agree on global structure?
  2. Which eigenvalue gap suggests the right embedding dimension?
  3. Are there visible clusters by year, language, or author?
  4. Next: distortion computation using saved Laplacian
""")


if __name__ == '__main__':
    main()

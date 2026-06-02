"""
run_eberle_embedding.py
-----------------------
Script to run megaman Spectral Embedding (Laplacian Eigenmaps / Diffusion Maps)
on Dr. Eberle's precomputed historical-document similarity matrices.

Data source: https://tubcloud.tu-berlin.de/s/oXCfSTgQjXQNZKd
  - similarity_gt.npz     : ground-truth similarity matrix
  - similarity_tfidf.npz  : cosine/TF-IDF precomputed similarity matrix
  - metadata.csv/.json    : per-document metadata (date, author, title, etc.)

Following Prof. Meila's advice:
  - Compute & save 10 eigenvectors + eigenvalues
  - Save the Laplacian matrix for distortion computation
  - Produce basic visualizations

Usage:
  python run_eberle_embedding.py --data_dir /path/to/downloaded/data
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, safe on any machine
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── megaman ──────────────────────────────────────────────────────────────────
from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_COMPONENTS   = 10        # number of eigenvectors to compute (per Prof. Meila)
LAPLACIAN_TYPE = 'geometric'   # options: 'geometric', 'symmetricnormalized',
                               # 'renormalized', 'randomwalk', 'unnormalized'
EIGEN_SOLVER   = 'arpack'      # robust default; try 'amg' for very large matrices
RANDOM_STATE   = 42

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_similarity_matrix(path: str):
    """
    Load a similarity matrix from .npz (scipy sparse), .npy (dense numpy),
    or .mat (MATLAB) format.  Returns a scipy CSR sparse matrix.
    """
    ext = os.path.splitext(path)[1].lower()
    print(f"  Loading {os.path.basename(path)} ...")

    if ext == '.npz':
        # Try scipy sparse first
        try:
            mat = sp.load_npz(path)
            print(f"    → sparse matrix  shape={mat.shape}  nnz={mat.nnz}")
            return mat.tocsr()
        except Exception:
            pass
        # Fall back to numpy archive (dense)
        arr = np.load(path, allow_pickle=True)
        key = list(arr.files)[0]
        mat = arr[key]
        print(f"    → dense array  shape={mat.shape}")
        return sp.csr_matrix(mat)

    elif ext == '.npy':
        mat = np.load(path, allow_pickle=True)
        print(f"    → dense array  shape={mat.shape}")
        return sp.csr_matrix(mat)

    elif ext == '.mat':
        data = sio.loadmat(path)
        # Find the first non-metadata key
        key = [k for k in data if not k.startswith('__')][0]
        mat = data[key]
        print(f"    → MATLAB matrix  key='{key}'  shape={mat.shape}")
        if sp.issparse(mat):
            return mat.tocsr()
        return sp.csr_matrix(mat)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def ensure_affinity_valid(A):
    """
    Megaman expects the affinity (similarity) matrix to be:
      - square, symmetric
      - non-negative (similarities, not distances!)
      - values in [0, 1] recommended; if values >> 1, normalize

    This function symmetrizes and clips negatives.
    """
    # Symmetrize
    A = 0.5 * (A + A.T)

    # Clip negatives (shouldn't happen with cosine / GT similarities)
    if sp.issparse(A):
        A.data = np.clip(A.data, 0, None)
    else:
        A = np.clip(A, 0, None)

    return A


def run_embedding(affinity_matrix, label: str, out_dir: str):
    """
    Given a precomputed affinity/similarity matrix, run Spectral Embedding:
      1. Build a Geometry object (bypassing adjacency/distance computation)
      2. Compute the Laplacian
      3. Compute eigenvectors / eigenvalues
      4. Save everything

    Returns
    -------
    embedding    : ndarray (n_docs, N_COMPONENTS)
    eigenvalues  : ndarray (N_COMPONENTS,)
    eigenvectors : ndarray (n_docs, N_COMPONENTS)
    laplacian    : sparse matrix (n_docs, n_docs)
    """
    print(f"\n{'='*60}")
    print(f"  Running embedding for: {label}")
    print(f"{'='*60}")

    # ── Step 1: Build Geometry, inject precomputed affinity ───────────────
    geom = Geometry(laplacian_method=LAPLACIAN_TYPE)
    geom.set_affinity_matrix(affinity_matrix)   # bypass adj/affinity computation

    # ── Step 2: Compute Laplacian (with full output for distortion later) ─
    print("  Computing Laplacian ...")
    laplacian = geom.compute_laplacian_matrix(copy=True, return_lapsym=True)
    print(f"    Laplacian type : {LAPLACIAN_TYPE}")
    print(f"    Laplacian shape: {laplacian.shape}")

    # ── Step 3: Spectral Embedding ────────────────────────────────────────
    print(f"  Running spectral decomposition (n_components={N_COMPONENTS}) ...")
    se = SpectralEmbedding(
        n_components  = N_COMPONENTS,
        geom          = geom,
        eigen_solver  = EIGEN_SOLVER,
        random_state  = RANDOM_STATE,
        drop_first    = True,   # drop trivial constant eigenvector
        diffusion_maps= False,
    )
    # Pass affinity directly — input_type='affinity' tells megaman not to recompute
    se.fit(affinity_matrix, input_type='affinity')

    embedding    = se.embedding_       # (n_docs, N_COMPONENTS)
    eigenvalues  = se.eigenvalues_     # (N_COMPONENTS,)
    eigenvectors = se.eigenvectors_    # (n_docs, N_COMPONENTS)
    laplacian    = se.laplacian_matrix_

    print(f"  Embedding shape : {embedding.shape}")
    print(f"  Eigenvalues     : {np.round(eigenvalues, 6)}")

    # ── Step 4: Save outputs ──────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, label)

    np.save(f"{prefix}_embedding.npy",    embedding)
    np.save(f"{prefix}_eigenvalues.npy",  eigenvalues)
    np.save(f"{prefix}_eigenvectors.npy", eigenvectors)

    # Save Laplacian as sparse NPZ (needed for distortion computation)
    sp.save_npz(f"{prefix}_laplacian.npz", sp.csr_matrix(laplacian))

    print(f"  Saved to: {out_dir}/")
    print(f"    {label}_embedding.npy    → (n_docs × {N_COMPONENTS})")
    print(f"    {label}_eigenvalues.npy  → ({N_COMPONENTS},)")
    print(f"    {label}_eigenvectors.npy → (n_docs × {N_COMPONENTS})")
    print(f"    {label}_laplacian.npz    → sparse Laplacian")

    return embedding, eigenvalues, eigenvectors, laplacian


def plot_embedding(embedding, label: str, out_dir: str, metadata=None):
    """
    Plot 2D scatter of the first two embedding dimensions.
    Optionally color by a metadata column (e.g., year, author).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Spectral Embedding — {label}", fontsize=14, fontweight='bold')

    # ── Panel 1: first two eigenvectors ───────────────────────────────────
    ax = axes[0]
    x, y = embedding[:, 0], embedding[:, 1]
    scatter = ax.scatter(x, y, s=15, alpha=0.7, c=np.arange(len(x)),
                         cmap='viridis', linewidths=0)
    plt.colorbar(scatter, ax=ax, label='Document index')
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')
    ax.set_title('Dims 1 & 2')

    # ── Panel 2: second and third eigenvectors ────────────────────────────
    ax = axes[1]
    if embedding.shape[1] >= 3:
        x2, y2 = embedding[:, 1], embedding[:, 2]
    else:
        x2, y2 = embedding[:, 0], embedding[:, 1]
    scatter2 = ax.scatter(x2, y2, s=15, alpha=0.7, c=np.arange(len(x2)),
                          cmap='plasma', linewidths=0)
    plt.colorbar(scatter2, ax=ax, label='Document index')
    ax.set_xlabel('Eigenvector 2')
    ax.set_ylabel('Eigenvector 3')
    ax.set_title('Dims 2 & 3')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{label}_embedding_2d.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def plot_eigenvalues(eigenvalues_dict: dict, out_dir: str):
    """Plot eigenvalue spectra for each similarity matrix."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, evals in eigenvalues_dict.items():
        ax.plot(np.arange(1, len(evals)+1), evals, 'o-', label=label, markersize=5)
    ax.set_xlabel('Eigenvector index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue spectra (larger = more informative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, 'eigenvalue_spectra.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Eigenvalue plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Megaman embedding for Dr. Eberle\'s data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the downloaded TU Berlin data files')
    parser.add_argument('--out_dir',  type=str, default='eberle_results',
                        help='Directory to save outputs (default: ./eberle_results)')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Auto-detect files in data_dir ────────────────────────────────────
    # Adjust these filenames to match whatever Dr. Eberle uploaded
    print("\nScanning data directory:", data_dir)
    all_files = os.listdir(data_dir)
    print("  Files found:", all_files)

    # Try to find the two similarity matrices — update names if needed
    # Common naming patterns; we'll try to be flexible:
    similarity_files = {}
    for f in all_files:
        fl = f.lower()
        if any(kw in fl for kw in ['gt', 'ground', 'groundtruth', 'ground_truth']):
            similarity_files['gt'] = os.path.join(data_dir, f)
        elif any(kw in fl for kw in ['tfidf', 'tf_idf', 'cosine', 'precomputed']):
            similarity_files['tfidf'] = os.path.join(data_dir, f)

    if not similarity_files:
        # Fallback: treat all matrix files as inputs
        for f in all_files:
            if f.endswith(('.npz', '.npy', '.mat')):
                key = os.path.splitext(f)[0]
                similarity_files[key] = os.path.join(data_dir, f)

    print(f"\nSimilarity matrices to process: {list(similarity_files.keys())}")

    # ── Run embedding for each matrix ─────────────────────────────────────
    eigenvalues_dict = {}

    for label, path in similarity_files.items():
        A = load_similarity_matrix(path)
        A = ensure_affinity_valid(A)

        embedding, evals, evecs, laplacian = run_embedding(A, label, out_dir)

        eigenvalues_dict[label] = evals
        plot_embedding(embedding, label, out_dir)

    # ── Eigenvalue spectra comparison ─────────────────────────────────────
    if eigenvalues_dict:
        plot_eigenvalues(eigenvalues_dict, out_dir)

    print("\n" + "="*60)
    print("  All done!  Results saved to:", out_dir)
    print("="*60)
    print("""
Next steps (for the meeting):
  1. Inspect the 2D plots — look for meaningful clusters
  2. Try coloring points by metadata (year, author, etc.)
  3. Examine eigenvalue gap to choose effective embedding dimension
  4. The saved Laplacian (.npz) can be used for distortion computation later
  5. Compare GT vs TF-IDF embeddings — do they agree on structure?
""")


if __name__ == '__main__':
    main()

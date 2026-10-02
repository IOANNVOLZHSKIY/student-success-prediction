from scipy import sparse

def to_csr(x):
    """Convert dense numpy array / DataFrame to CSR sparse matrix."""
    return sparse.csr_matrix(x)
import numpy as np 


def summzarize_split(X, name = 'sparse_matix'):
    # basic validity guard (sparse-like)
    if not hasattr(X, 'shape') or not hasattr(X, 'nnz'):
        raise ValueError(f"Input data for {name} is not a valid sparse matrix.")

    n_docs, n_feats = X.shape
    nnz_rows = X.getnnz(axis=1)          # non-zeros per document
    nnz_cols = X.getnnz(axis=0)          # non-zeros per feature
    n_all_zero_feats = (nnz_cols == 0).sum()
    n_empty_docs = (nnz_rows == 0).sum()
    density = X.nnz / (n_docs * n_feats) #  density

    print(f'The number of Documents in {name} split is {n_docs}')
    print(f'The number of Features in {name} split is {n_feats}')
    print(f'Number of all-zero features in {name} split is {n_all_zero_feats}')
    print(f'The Density of the {name} split is {density:.6f}')
    print(f'Number of non-zero entries per row in {name} is {nnz_rows}')
    print('---------------------------------')
    print(f'The Total Number of non-zero entries in {name} is {nnz_rows.sum()}')
    print(f'Empty rows (docs with no surviving features) is {n_empty_docs}')
    print(f'Percentage of empties is {n_empty_docs / n_docs:.4%}')
    print(f"Mean nnz per doc: {nnz_rows.mean():.4f}")
    print(f"Median nnz per doc: {np.median(nnz_rows):.4f}")
    print(f"95th percentile nnz per doc: {np.percentile(nnz_rows, 95):.2f}")



def vectorizer_summary(vec):
    params = {}
    # common
    params["ngram_range"]  = getattr(vec, "ngram_range", None)
    params["min_df"]       = getattr(vec, "min_df", None)
    params["max_df"]       = getattr(vec, "max_df", None)
    params["max_features"] = getattr(vec, "max_features", None)
    params["lowercase"]    = getattr(vec, "lowercase", None)
    params["stop_words"]   = getattr(vec, "stop_words", None)
    # tfidf-specific
    if hasattr(vec, "use_idf"):
        params["use_idf"]      = vec.use_idf
        params["sublinear_tf"] = getattr(vec, "sublinear_tf", None)
        params["norm"]         = getattr(vec, "norm", None)

    for k, v in params.items():
        print(f"\n{k:>14}:  ======> s{v}")

    # Vocabulary
    feature_names = vec.get_feature_names_out()
    V = len(feature_names)
    print(f"\nVocabulary size:  ======>  {V:,}")

    if V:
        print('\nSample of the Features: ')
        print(', '.join(vec.get_feature_names_out()[:10]), ',,,', ', '.join(vec.get_feature_names_out()[-5:]))
    
    if hasattr(vec, 'idf_'):
        idf = vec.idf_
        print(f"\nIDF range: min = {idf.min():.4f}, Median = {np.median(idf):.4f}, Max = {idf.max(): .4f}")

    # rarest words (high IDF) and common words (Los IDF)
        feat = vec.get_feature_names_out()
        hi = np.argsort(idf)[::-1][:20]
        low = np.argsort(idf)[:20]
        print ('#######################################')


        print(f"The Top 20 terms by IDF (Rarest) are :")
        print(', '.join(feat[hi]))

        print ('#######################################')
        print(f"The Top 20 terms by LOW IDF (Common) are :")
        print(', '.join(feat[low]))
    else:
        print("\n IDF stats not available for CounVectorizer")

    print ('\n=== END OF SUMMARY ===')
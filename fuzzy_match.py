import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

def fuzzy_match(
        df1,
        df2, 
        left_on, 
        right_on, 
        similarity_threshold, 
        how, 
        ngram_range, 
        top_n, 
        indicator = "_merge", 
        cos_similarity_score = True
    ):

    """
    Perform fuzzy matching between two DataFrames using cosine similarity on specified columns.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - left_on (list of str): Columns in df1 to match on.
    - right_on (list of str): Columns in df2 to match on.
    - similarity_threshold (list of float): Minimum cosine similarity score for each matching column.
    - how (str): Type of merge (e.g., 'left', 'right', 'inner', 'outer').
    - ngram_range (tuple of int): N-gram range for TF-IDF vectorizer.
    - top_n (int): Maximum number of matches to return per row.
    - indicator (str): Column name to indicate the merge status. Defaults to "_merge". If indicator = False, no indicator column will be added. If indicator = True, the default name "_merge" will be used.
    - cos_similarity_score (bool, optional): Whether to include cosine similarity scores in the resulting dataframe. Defaults to True.

    Returns:
    - pd.DataFrame: The merged DataFrame with fuzzy matched rows.
    """

    all_matches = pd.DataFrame()
    matches_df_dict = {}

    # Initialize the vectorizer outside the loop
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)

    for i in range(len(left_on)):
        left_col = left_on[i]
        right_col = right_on[i]
        threshold = similarity_threshold[i]

        # Check for empty merge keys
        if df1[left_col].isna().all() or df2[right_col].isna().all():
            matches_df = pd.DataFrame(columns=[
                'df1_index', 'df2_index', f'cos_similarity_{i}'
            ])
        else:
            # Convert to tf-idf vectors
            tfidf_left = vectorizer.fit_transform(df1[left_col].fillna("").astype(str))
            tfidf_right = vectorizer.transform(df2[right_col].fillna("").astype(str))

            # Perform sparse matrix multiplication for fast cosine similarity
            matches = sp_matmul_topn(tfidf_left, tfidf_right.T, top_n=top_n, threshold=threshold, sort=True)
            left_indices, right_indices = matches.nonzero()

            match_data = {
                'df1_index': left_indices,
                'df2_index': right_indices,
                f'cos_similarity_{i}': matches.data
            }
            matches_df = pd.DataFrame(match_data)

        # Combine results across multiple merge keys
        if i == 0:
            all_matches = matches_df
        else:
            all_matches = all_matches.merge(matches_df, on=['df1_index', 'df2_index'], how='inner')

        matches_df_dict[i] = matches_df

    # Merge df1 with df2 based on the matched indices
    df1_merged = df1.reset_index(drop=True).merge(all_matches, left_index=True, right_on='df1_index', how='left')
    final_df = df1_merged.merge(df2, left_on='df2_index', right_index=True, how=how, indicator=indicator)

    # Drop temporary index columns
    final_df.drop(columns=['df1_index', 'df2_index'], inplace=True)

    # Optionally remove cos_similarity_i columns
    if not cos_similarity_score:
        cos_cols = [col for col in final_df.columns if col.startswith("cos_similarity_")]
        final_df.drop(columns=cos_cols, inplace=True)

    # Reorder columns: move merge indicator and cos_similarity_i to the end
    cols = final_df.columns.tolist()
    indicator_cols = [indicator] if indicator in cols else []
    similarity_cols = [col for col in cols if col.startswith("cos_similarity_")]

    # Create the final column order
    core_cols = [col for col in cols if col not in indicator_cols + similarity_cols]
    final_order = core_cols + indicator_cols + similarity_cols

    # Apply the new column order
    final_df = final_df[final_order].reset_index(drop=True)

    return final_df

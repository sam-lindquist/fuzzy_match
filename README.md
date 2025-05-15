# fuzzy_match
This function uses cosine similarity scoring from [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn) to fuzzy match data sets in Python. Calculating cosine similarity scores is known to be faster than calculating other types of string metrics. 

### Features
- Merge on multiple variables across your datasets
- Set minimum allowable similarity scores for each variable (with the `similarity_threshold` argument)
- Set how many matches you are willing to allow for a given string with `top_n` (e.g., you could keep only the "top" match, or could keep more)

### Arguments
The function takes in as parameters: 
- `df1` (pd.DataFrame): First DataFrame.
- `df2` (pd.DataFrame): Second DataFrame.
- `left_on` (list of str): Columns in df1 to match on.
- `right_on` (list of str): Columns in df2 to match on.
- `similarity_threshold` (list of float): Minimum cosine similarity score for each matching column.
- `how` (str): Type of merge (e.g., 'left', 'right', 'inner', 'outer').
- `ngram_range` (tuple of int): N-gram range for TF-IDF vectorizer.
- `top_n` (int): Maximum number of matches to return per row.
- `indicator` (str): Column name to indicate the merge status. Defaults to "_merge". If indicator = False, no indicator column will be added. If indicator = True, the default name "_merge" will be used.
- `cos_similarity_score` (bool, optional): Whether to include cosine similarity scores in the resulting dataframe. Defaults to True.

It then returns a pd.DataFrame with fuzzy matched rows.

### Example usage

```
from fuzzy_match import fuzzy_match
import pandas as pd 

df1 = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
df2 = pd.DataFrame({"name": ["Alicia", "Bob", "Charles"]})

result = fuzzy_match(
    df1, df2,
    left_on=["name"],
    right_on=["name"],
    similarity_threshold=[0.6],
    how="left",
    indicator="_merge",
    ngram_range=(2, 3),
    top_n=1
)

print(result)
```

### Dependencies
The function is dependent upon:
- sp_matmul_topn from [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn)
- TfidfVectorizer from [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [pandas](https://github.com/pandas-dev/pandas)

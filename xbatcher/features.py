"""Functions for transforming xarray datasets into features that can
be input to machine learning libraries."""

def dataset_to_feature_dataframe(ds, coords_as_features=False):
    df = ds.to_dataframe()
    return df

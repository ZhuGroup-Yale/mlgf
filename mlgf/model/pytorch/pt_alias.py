from mlgf.model.pytorch.gnn import GNN

def get_model_from_alias(alias, **kwargs):
    """add your own models here as you add them in the model.pytorch folder

    Args:
        alias (str): name of model

    Returns:
        GNN
    """    
    if alias == 'GNN':
        return GNN(**kwargs)
   
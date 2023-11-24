from .csfever import CSFEVERDataset
from .fever import FEVERDataset
from .dataset import Dataset
from .ctkfacts import CTKFactsDataset
from .xfact import XFactDataset
from .demagog import DemagogDataset
from .liar import LiarDataset
from .multiclaim import MultiClaimDataset
from .multifc import MultiFCDataset
from .afp import AFPDataset
from .slovaksum import SlovakSumDataset
from .squadsk import SquadSKDataset
from .ud import UDDataset
from .wikipedia import WikipediaDataset


def get_dataset(dataset_name: str, path: str, language:str) -> Dataset:
    if dataset_name == 'fever':
        return FEVERDataset(path)
    elif dataset_name == 'csfever':
        return CSFEVERDataset(path)
    elif dataset_name == 'ctkfacts':
        return CTKFactsDataset(path)
    elif dataset_name == 'xfact':
        return XFactDataset(path)
    elif dataset_name == 'demagog':
        return DemagogDataset(path, language=language)
    elif dataset_name == 'liar':
        return LiarDataset(path)
    elif dataset_name == 'multiclaim':
        return MultiClaimDataset(path)
    elif dataset_name == 'multifc':
        return MultiFCDataset(path)
    elif dataset_name == 'afp':
        return AFPDataset(path)
    elif dataset_name == 'slovaksum':
        return SlovakSumDataset(path)
    elif dataset_name == 'squadsk':
        return SquadSKDataset(path)
    elif dataset_name == 'ud':
        return UDDataset(path, language=language)
    elif dataset_name == 'wikipedia':
        return WikipediaDataset(path, language=language)
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')

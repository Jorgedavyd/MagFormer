#Lighting utils for data pipeline automation
from starstream import Callable, datetime_interval
from lightning import LightningDataModule

# General utils for Dataset creation
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as tt

# Data dependencies
from corkit import lasco
from starstream.downloaders import SOHO, WIND, ACE, DSCOVR, SWARM, SDO, Dst

#Other general utils
from datetime import timedelta, datetime
import pandas as pd
import torch
from typing import Dict, List, Tuple, Sequence
from torch import Tensor
from PIL import Image
import os.path as osp
import glob
"""
base_scrap_date: Means the scrap date of the Dst index from which
we are downloading the historic data. We create a relative scrap
date based on this one for prediction purposes.

relative_scrap_date: base_scrap_date shifted within the range of
2 hours for forecasting purposes.
"""
# Utils for data synchronization

## Compute the sequence length from the datset of input date scrap_date
def dataset_len(scrap_date: Tuple[datetime, datetime], dst_sl: timedelta):
    delta_t = scrap_date[-1] - scrap_date[0]
    return (delta_t - dst_sl)/timedelta(hours = 1)

# Sets the data index relative to the dst index
class DstBasedIdx:
    @staticmethod
    def L1Idx(idx: int, dst_init_date: datetime, l1_sl: timedelta):
        d_0 = dst_init_date - l1_sl
        return slice(pd.Timestamp(d_0 + idx*timedelta(hours = 1)),pd.Timestamp(d_0 + l1_sl+(idx*timedelta(hours = 1))))
    @staticmethod
    def SunIdx(idx: int, dst_init_date: datetime,sl: timedelta):
        d_0 = dst_init_date - sl
        return slice(pd.Timestamp(d_0 + idx*timedelta(hours = 1)), pd.Timestamp(d_0 + sl + (idx*timedelta(hours = 1))))
    @staticmethod
    def SWARMIdx(idx: int, dst_init_date: datetime):
        return slice(pd.Timestamp(dst_init_date + idx*timedelta(hours = 1)), pd.Timestamp(dst_init_date+(idx+1)*timedelta(hours = 1)))
    @staticmethod
    def DstIdx(idx: int, dst_init_date: datetime, dst_sl: timedelta):
        return slice(pd.Timestamp(dst_init_date + idx*timedelta(hours = 1)), pd.Timestamp(dst_init_date+idx*timedelta(hours = 1)+ dst_sl))
    @staticmethod
    def __call__(idx: int, dst_init_date: datetime, l1_sl: timedelta, cme_sl: timedelta):
        cme_idx = DstBasedIdx.SunIdx(idx, dst_init_date, cme_sl)
        l1_idx = DstBasedIdx.L1Idx(idx, dst_init_date, l1_sl)
        swarm_idx = DstBasedIdx.SWARMIdx(idx, dst_init_date)
        dst_idx = DstBasedIdx.DstIdx(idx, dst_init_date)
        return l1_idx, cme_idx, swarm_idx, dst_idx

# Sets the scrap index relative to the dst index scrap parameters
class DstBasedScrap:
    @staticmethod
    def SWARM(scrap_date: tuple[datetime, datetime], dst_sl: timedelta):
        return scrap_date[0], scrap_date[-1] + dst_sl
    @staticmethod
    def Dst(scrap_date: tuple[datetime, datetime], dst_sl: timedelta):
        return scrap_date[0], scrap_date[-1] + dst_sl
    @staticmethod
    def Sun(scrap_date: tuple[datetime, datetime], sl:timedelta):
        return scrap_date[0] - sl, scrap_date[-1]
    @staticmethod
    def L1(scrap_date: tuple[datetime, datetime], l1_sl: timedelta):
        return scrap_date[0] - l1_sl, scrap_date[-1]

class L1DatasetLevel1(Dataset):
    def __init__(
            self,
            dataset: pd.DataFrame,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
            step_size: timedelta,
            l1: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self.len_dataset = dataset_len(scrap_date, dst_sl)
        self.idx = lambda idx: DstBasedIdx.L1Idx(idx, scrap_date[0], sequence_length)
        self.step_size = step_size
        self.scrap_date_list = DstBasedScrap.L1(scrap_date, sequence_length)
        self.l1 = l1
        self.dataset = dataset
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        raise NotImplementedError('Not implemented module')
    def __len__(self):
        return self.len_dataset
    def __getitem__(self, idx) -> Tensor:
        return torch.from_numpy(self.dataset[self.idx(idx)].values)

class DatasetLevel2(Dataset):
    def __init__(self, base_class, base_scrap_date_list: List[Tuple[datetime, datetime]], step_size: timedelta) -> None:
        self.datasets = [base_class(scrap_date, step_size) for scrap_date in base_scrap_date_list]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx) -> Tensor:
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while idx >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])
        if dataset_idx > 0:
            idx -= sum(len(self.datasets[i]) for i in range(dataset_idx))

        return self.datasets[dataset_idx][idx]

class DatasetLevel3(Dataset):
    def __init__(self, *args) -> None:
        self.datasets = args
    def __len__(self) -> int:
        return len(self.datasets[0])
    def __getitem__(self, idx) -> Tuple[Tensor, ...]:
        return tuple([dataset[idx] for dataset in self.datasets])

class WINDDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [WIND.TDP_PM(), WIND.MAG()]
        dataset =  list(map(lambda x: x.data_prep(scrap_date, step_size), dataset))
        return pd.concat(dataset, axis = 1)

class ACEDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [ACE.SWEPAM(), ACE.MAG(), ACE.SIS(), ACE.EPAM()]
        dataset =  list(map(lambda x: x.data_prep(scrap_date, step_size), dataset))
        return pd.concat(dataset, axis = 1) ## TODO REVISAR LOS PREPS

class DSCOVRDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        return DSCOVR().data_prep(scrap_date, step_size)

class SOHODataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [SOHO.CELIAS_PM(), SOHO.CELIAS_SEM()]
        dataset =  list(map(lambda x: x.data_prep(scrap_date, step_size), dataset))
        return pd.concat(dataset, axis = 1) ## TODO REVISAR LOS PREPS

class DstDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        return Dst().data_prep(scrap_date, step_size)

class SwarmDataset(L1DatasetLevel1):
    def __init__(
            self,
            dataset: pd.DataFrame,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
            step_size: timedelta,
            l1: bool = True,
    ) -> None:
        super().__init__(dataset, scrap_date, sequence_length, dst_sl, step_size, l1)
        self.idx = lambda idx: DstBasedIdx.SWARMIdx(idx, scrap_date[0])
        self.scrap_date_list = DstBasedScrap.SWARM(scrap_date, sequence_length)
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [
            SWARM.MAG_ION(scrap_date, step_size),
            SWARM.EFI(scrap_date, step_size),
            SWARM.FAC(scrap_date, step_size),
        ]
        dataset =  list(map(lambda x: x.data_prep(scrap_date, step_size), dataset))
        return pd.concat(dataset, axis = 1) ## TODO REVISAR LOS PREPS

class _ImageryBase(Dataset):
    def __init__(
            self,
            root_path: str,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ) -> None:
        self.root_path: str = root_path
        self.path_to_image: Callable[[str], str] = lambda name: osp.join(self.root_path, f"{name}.png")
        self.raw_dataset: List[str] = glob.glob(self.path_to_image('*'))
        self.sequence_length = sequence_length
        self.len_dataset = dataset_len(scrap_date, dst_sl)
        self.idx = lambda idx: DstBasedIdx.SunIdx(idx, scrap_date[0], sequence_length)
        self.scrap_date_list = DstBasedScrap.Sun(scrap_date, sequence_length)
        interval_time: List[str] = datetime_interval(*self.scrap_date_list, timedelta(days = 1))
        self.data: Dict[str, List[str]] = {datetime.strptime(key, '%Y%m%d'): [path for path in self.raw_dataset if key in path] for key in interval_time}
        self.transforms: tt.Compose = tt.Compose([
            tt.PILToTensor(),
            tt.Resize(256),
            tt.Normalize() ## setup the values
        ])
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx: int | slice) -> Tensor:
        idx = self.idx(idx)
        img_paths: List[List[str]] = list(self.data.values())[idx]
        return torch.cat([
            self.transforms(Image.open(path)) for list_paths in img_paths for path in list_paths
        ], dim = 1)

# Dataset for all tasks involved
class LASCODataset(_ImageryBase):
    def __init__(
            self,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ):
        super().__init__('./data/SOHO/LASCO/', scrap_date, sequence_length, dst_sl)

# Dataset for all tasks involved
class SDODataset(_ImageryBase):
    def __init__(
            self,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ):
        super().__init__('./data/SDO/AIA_HR/', scrap_date, sequence_length, dst_sl)


# Dataset for all tasks involved
"""
Supervised synthetic data creation:
model: SOHO, WIND ----> DSCOVR, ACE
"""
class DefaultDataModule(LightningDataModule):
    def __init__(self, dataset: DatasetLevel3, batch_size: int, num_workers: int, pin_memory: bool, train_p: float) -> None:
        super().__init__()
        self.dataset = dataset
        self.train_p = train_p
        self.batch_size = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: int = pin_memory

    def setup(self) -> None:
        # Getting the random_split set
        train_len = round(len(self) * self.train_p)
        val_len = (len(self.dataset) - train_len)//2
        test_len = len(self.dataset) - (train_len + val_len)
        # Random split on datasets
        train_ds, val_ds, test_ds = random_split(self.dataset, [train_len, val_len, test_len])
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            True,
            num_workers = 12,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size * 2,
            False,
            num_workers = 12,
            pin_memory=True
        )

    def test_dataloder(self):
        return DataLoader(
            self.test_ds,
            self.batch_size,
            True,
            num_workers = 12,
            pin_memory=True
        )

#Lighting utils for data pipeline automation
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
from typing import List, Tuple, Sequence
from torch import Tensor

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
    def __getitem__(self, idx) -> Sequence[Tensor]:
        return [dataset[idx] for dataset in self.datasets]

class WINDDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [WIND.TDP_PM(), WIND.MAG()]
        dataset = list(map(dataset, lambda x: x.data_prep(scrap_date, step_size)))
        return pd.concat(dataset, axis = 1)

class ACEDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [ACE.SWEPAM(), ACE.MAG(), ACE.SIS(), ACE.EPAM()]
        dataset =  list(map(dataset, lambda x: x.data_prep(scrap_date, step_size)))
        return pd.concat(dataset, axis = 1) ## TODO REVISAR LOS PREPS

class DSCOVRDataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        return DSCOVR().data_prep(scrap_date, step_size)

class SOHODataset(L1DatasetLevel1):
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        dataset = [SOHO.CELIAS_PM(), SOHO.CELIAS_SEM()]
        dataset =  list(map(dataset, lambda x: x.data_prep(scrap_date, step_size)))
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
        self.idx = lambda idx: DstBasedIdx.SWARMIdx(idx, scrap_date[0], sequence_length)
        self.scrap_date_list = DstBasedScrap.SWARM(scrap_date, sequence_length)
    def prepare_data(self, scrap_date, step_size) -> pd.DataFrame:
        # data prep
        dataset = [
            SWARM.MAG_ION(scrap_date, step_size),
            SWARM.EFI(scrap_date, step_size),
            SWARM.FAC(scrap_date, step_size),
        ]
        dataset =  list(map(dataset, lambda x: x.data_prep(scrap_date, step_size)))
        return pd.concat(dataset, axis = 1) ## TODO REVISAR LOS PREPS

class _ImageryBase(Dataset):
    def __init__(
            self,
            root_path: str,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ) -> None:
        self.sequence_length = sequence_length
        self.len_dataset = dataset_len(scrap_date, dst_sl)
        self.idx = lambda idx: DstBasedIdx.SunIdx(idx, scrap_date[0], sequence_length)
        self.scrap_date_list = DstBasedScrap.Sun(scrap_date, sequence_length)
    def __len__(self):
        return self.len_dataset
    def __getitem__(self, idx) -> Tensor:
        return torch.from_numpy(self.dataset[self.idx(idx)].values)

# Dataset for all tasks involved
class LASCODataset(_ImageryBase):
    def __init__(
            self,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ):
        super().__init__(root_path, scrap_date, sequence_length, dst_sl)

# Dataset for all tasks involved
class SDODataset(_ImageryBase):
    def __init__(
            self,
            scrap_date: Tuple[datetime, datetime],
            sequence_length: timedelta,
            dst_sl: timedelta,
    ):
        super().__init__(root_path, scrap_date, sequence_length, dst_sl)


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
    def setup(self) -> None:
        # Getting the random_split set
        train_len = round(len(self) * self.train_p)
        val_len = (len(self) - train_len)//2
        test_len = len(self) - (train_len + val_len)
        # Random split on datasets
        train_ds, val_ds, test_ds = random_split(self.dataset, [train_len, val_len, test_len])
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        #Done

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

class Reconstruction:
    """
    # L1 RECONSTRUCTION
    model: corrupted(DSCOVR, ACE) -> DSCOVR, ACE
    """
    class L1Reconstruction(DataModule):
        """
        This is for training the VAE that reconstructs
        the anomaly data for better results.
        """
        def __init__(
                self,
                base_scrap_date_list: List[datetime],
                l1_sl: timedelta,
                dst_sl: timedelta,
                step_size: timedelta,
                batch_size: timedelta,
                train_p: float
        ) -> None:
            super().__init__(batch_size, train_p)
            self.l1_sl = l1_sl
            self.dst_sl = dst_sl
            self.step_size = step_size
            self.base_scrap_date_list = base_scrap_date_list

            for scrap_date in base_scrap_date_list:
                if scrap_date[0] < datetime(2016, 7, 31) or scrap_date[-1] > datetime(2021, 10, 31):
                    raise ValueError('Cannot train on this range given the data access')

        def prepare_data(self, scrap_date, step_size) -> None:
            ## this training will be valid for the current intervals 2016 -> 2021
            self.l1_ace = ACEDataset(base_scrap_date_list, l1 = True)
            self.l1_dscovr = DSCOVRDataset(base_scrap_date_list, l1 = True)
            self.l2_dscovr = DSCOVRDataset(base_scrap_date_list)
            self.l2_ace = ACEDataset(base_scrap_date_list)
            self.dataset = MultiDataset(
                ACEDataset()
            )
            assert (len(self.l1_ace) == len(self.l2_ace) == len(self.l2_dscovr) == len(self.l1_dscovr)), 'Invalid datasets debug']

        def __getitem__(self, index) -> Tuple[Tensor, ...]:
            return (
                self.l1_ace[index],
                self.l1_dscovr[index],
                self.l2_ace[index],
                self.l2_dscovr[index],
            )

        def __len__(self):
            return sum([len(dataset) for dataset in self.dataset])

    class ImageReconstruction(DataModule):
        """
        For both LASCO and SDO, and whatever imagery data
        you have.
        """

        corrupted = tt.Compose([
            tt.ToTensor(),
            tt.RandomErasing(p = 1)
        ])

        normal = tt.Compose([
            tt.ToTensor(),
        ])

        def __init__(
                self,
                base_scrap_date_list: List[datetime],
                dataset,
                batch_size: int,
                train_p: float
        ) -> None:
            super().__init__(batch_size, train_p)
            self.dataset = dataset
            self.base_scrap_date_list = base_scrap_date_list

        def prepare_data(self) -> None:
            self.dataset = self.dataset(self.base_scrap_date_list)
            self.dataset.prepare_data()

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
            x_hat = self.corrupted(index)
            x = self.normal(index)
            return x_hat, x

# TODO: Review this code
# TODO: Define carefully all of the data_prep members

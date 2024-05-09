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
from typing import List, Tuple, Dict, Callable
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
def dataset_len(scrap_date, dst_sl: timedelta):
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

class _Base(Dataset):
    def __len__(self):
        return self.len_dataset
    def __getitem__(self, idx) -> Tensor:
        return self.dataset[self.idx(idx)]

## Datasets for satellites
class L1Base(_Base):
    def __init__(
            self,
            base_scrap_date: List[Tuple[datetime, datetime]],
            sequence_length: timedelta,
            dst_sl: timedelta,
            step_size: timedelta,
            l1: bool = True,
    ) -> None:
        self.sequence_length = sequence_length
        self.len_dataset = dataset_len(base_scrap_date, dst_sl)
        self.idx = lambda idx: DstBasedIdx.L1Idx(idx, base_scrap_date[0], sequence_length)
        self.step_size = step_size
        self.scrap_date_list = DstBasedScrap.L1(base_scrap_date, sequence_length)
        self.l1 = l1

class ACEDataset(L1Base, ACE):
    def __init__(self, base_scrap_date: List[Tuple[datetime]], sequence_length: timedelta, dst_sl: timedelta, step_size: timedelta, l1: bool = True) -> None:
        super().__init__(base_scrap_date, sequence_length, dst_sl, step_size, l1)
    
    def prepare_data(self):
        # download -> data prep -> torch

class DSCOVRDataset(L1Base, DSCOVR):
    def __init__(self, base_scrap_date: List[Tuple[datetime]], sequence_length: timedelta, dst_sl: timedelta, step_size: timedelta, l1: bool = True) -> None:
        super().__init__(base_scrap_date, sequence_length, dst_sl, step_size, l1)
    
    def prepare_data(self):
        # download -> data prep -> torch



class DstDataset(_Base, Dst):
    def __init__(
            self,
            base_scrap_date: tuple[datetime, datetime],
            dst_sl: timedelta,
    ):
        self.idx: Callable[[int], slice]= lambda idx:  DstBasedIdx.DstIdx(idx, base_scrap_date[0], dst_sl)
        self.sl: timedelta = dst_sl 
        self.scrap_date: Tuple[datetime] = DstBasedScrap.Dst(base_scrap_date, dst_sl)

    def prepare_dataset(self) -> None:
        # download -> data prep -> torch
        self.dst = self.data_prep(self.scrap_date)
        
class SwarmDataset(_Base, SWARM):
    def __init__(
            self,
            base_scrap_date: tuple[datetime,datetime],
            step_size: timedelta,
            dst_sl: timedelta,
    ):
        # Compute the index based on the relative time
        self.idx = lambda idx: DstBasedIdx.SWARMIdx(idx,base_scrap_date[0], dst_sl)
        self.len_dataset = dataset_len(base_scrap_date, dst_sl)
        self.sl = dst_sl
        self.scrap_date = DstBasedScrap.SWARM(base_scrap_date, dst_sl)
        self.step_size = step_size
    
    def prepare_data(self) -> None:
        # download -> data_prep -> tensor
        
        # data prep
        self.dataset = [
            self.MAG_ION().data_prep(self.scrap_date, self.step_size),
            self.EFI().data_prep(self.scrap_date, self.step_size),
            self.FAC().data_prep(self.scrap_date, self.step_size),
        ]
        self.dataset = torch.from_numpy(pd.concat(self.dataset, axis = 1).values)
    
class _ImageryBase(_Base):
    def __init__(
            self,
            base_scrap_date: tuple[datetime,datetime],
            step_size: timedelta,
            dst_sl: timedelta,

    ):
        # Compute the index based on the relative time
        self.idx = lambda idx: DstBasedIdx.SunIdx(idx,base_scrap_date[0], dst_sl)
        self.len_dataset = dataset_len(base_scrap_date, dst_sl)
        self.sl = dst_sl
        self.scrap_date = DstBasedScrap.Sun(base_scrap_date, dst_sl)
        self.step_size = step_size

    def prepare_data(self, idx) -> None:
        # download -> prepare -> return
    
# Dataset for all tasks involved
class LASCODataset(_ImageryBase):
    def __init__(
            self,
            base_scrap_date: tuple[datetime,datetime],
            step_size: timedelta,
            dst_sl: timedelta,

    ):
        super().__init__(base_scrap_date, step_size, dst_sl)
    def prepare_data(self, idx) -> None:
        # download -> prepare -> return

class SDODataset(_ImageryBase, SDO):
    def __init__(
            self,
            base_scrap_date: tuple[datetime,datetime],
            step_size: timedelta,
            dst_sl: timedelta,

    ):
        super().__init__(base_scrap_date, step_size, dst_sl)
    def prepare_data(self, idx) -> None:
        # download -> prepare -> return
        


# Dataset for all tasks involved

"""
Supervised synthetic data creation:
model: SOHO, WIND ----> DSCOVR, ACE
"""
class DataModule(LightningDataModule):

    def __init__(self, batch_size: int, train_p: float) -> None:
        super().__init__()
        self.train_p = train_p
        self.batch_size = batch_size

    def setup(self) -> None:
        # Getting the random_split set
        train_len = round(len(self) * self.train_p)
        val_len = (len(self) - train_len)//2
        test_len = len(self) - (train_len + val_len)
        # Random split on datasets
        train_ds, val_ds, test_ds = random_split(self, [train_len, val_len, test_len])
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
    

class SyntheticTask(DataModule):
    def __init__(self, batch_size: int, train_p: float) -> None:
        super().__init__(batch_size, train_p)
        # 2015 -> 2024

    def prepare_data(self, Dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, ...]:

    def __len__(self):
        

class MultiDataset(Dataset):
    def __init__(self, *datasets) -> None:
        self.datasets = datasets
    
    def __getitem__(self, index) -> Tuple[Tensor, ...]:
        return (dataset[index] for dataset in self.datasets)    


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

        def prepare_data(self) -> None:
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
        

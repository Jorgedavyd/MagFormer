#Lighting utils for data pipeline automation
from lightning import LightningDataModule

# General utils for Dataset creation
from torch.utils.data import Dataset, DataLoader, random_split

# Data dependencies
from corkit import lasco
from starstream.downloaders import SOHO, WIND, ACE, DSCOVR, Dst

#Other general utils
from datetime import timedelta, datetime
import pandas as pd
import torch


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
        cme_idx = DstBasedIdx.CMEsIdx(idx, dst_init_date, cme_sl)
        l1_idx = DstBasedIdx.L1Idx(idx, dst_init_date, l1_sl)
        flare_idx = DstBasedIdx.FlareIdx(idx, dst_init_date)
        swarm_idx = DstBasedIdx.SWARMIdx(idx, dst_init_date)
        dst_idx = DstBasedIdx.DstIdx(idx, dst_init_date)
        return flare_idx, l1_idx, cme_idx, swarm_idx, dst_idx

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
    
# Datasets for all Data types and satellites used
class Target:
    class DstDataset(Dataset):
        def __init__(
                self,
                base_scrap_date: tuple[datetime, datetime],
                dst_sl: timedelta,
        ):
            self.idx = lambda idx:  DstBasedIdx.DstIdx(idx, base_scrap_date[0], dst_sl)
            self.sl = dst_sl 
            self.dst = Dst().data_prep(base_scrap_date)

        def __len__(self):
            return len(self.dst)
        def __getitem__(self, idx):
            idx = self.idx(idx)
            return torch.from_numpy(self.dst.loc[idx].to_pandas().values)

class Tabular:
    class ACE(Dataset):

    class WIND(Dataset):

    class SOHO(Dataset):

    class DSCOVR(Dataset):

class Imagery:
    # LASCO Dataset for most important events
    class LASCODataset(Dataset):

    # SDO AIA Dataset for most important events
    class EUVDataset(Dataset):

# Dataset for all tasks involved

"""
Supervised synthetic data creation:
model: SOHO, WIND ----> DSCOVR, ACE
"""
class SyntheticTask:

class Reconstruction:
    """
    # L1 RECONSTRUCTION
    model: corrupted(DSCOVR, ACE) -> DSCOVR, ACE
    """
    class L1Reconstruction()
        
    class OMNIReconstruction()
    
    class EUVReconstruction()
        
    class LASCOReconstruction()
    

    
class DataModule(LightningDataModule):

    def __init__(self, dataset: Dataset, batch_size: int, train_p: float) -> None:
        super().__init__()
        self.dataset = dataset
        self.train_p = train_p
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        

    def setup(self) -> None:
        # Getting the random_split set
        train_len = round(len(self.dataset) * self.train_p)
        val_len = (len(self.dataset) - train_len)//2
        test_len = len(self.dataset) - (train_len + val_len)
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
    
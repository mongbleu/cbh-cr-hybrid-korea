import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor

class SamplingImsang:
    def __init__(self, imsang_df, attribute_list = ['DNST_CD', 'DMCLS_CD', 'HEIGHT'],
                 h_dict = None, dbh_dict = None, cd_dict = None,
                 h_samples = None, dbh_samples = None, cd_samples = None):
        self.imsang_df = imsang_df
        self.attribute_list = attribute_list
        self.h_dict = h_dict
        self.dbh_dict = dbh_dict
        self.cd_dict = cd_dict
        self.h_samples = h_samples
        self.dbh_samples = dbh_samples
        self.cd_samples = cd_samples
        
# ===== Sampling Function by Rows =====
    def sample_attributes(self, idx_data):
        idx, data = idx_data
        result = [np.nan] * len(self.attribute_list)
        try:
            cd, dbh, h = data
            if (cd.strip() == '') or (dbh.strip() == '') or (h.strip() == ''):
                return idx, result

            # HEIGHT
            if h.strip():
                lower, upper = self.h_dict[h]
                filtered = self.h_samples[(self.h_samples > lower) & (self.h_samples <= upper)]
                attempts = 0
                while (len(filtered) < 20) and (attempts < 10):
                    lower = max(0, lower - 1)
                    upper = min(40, upper + 1)
                    filtered = self.h_samples[(self.h_samples > lower) & (self.h_samples <= upper)]
                    attempts += 1
                result[0] = np.random.choice(filtered)

            # DBH
            if dbh.strip():
                lower, upper = self.dbh_dict[dbh]
                filtered = self.dbh_samples[(self.dbh_samples > lower) & (self.dbh_samples <= upper)]
                attempts = 0
                while (len(filtered) < 20) and (attempts < 10):
                    lower = max(0, lower - 6)
                    upper = min(110, upper + 6)
                    filtered = self.dbh_samples[(self.dbh_samples > lower) & (self.dbh_samples <= upper)]
                    attempts += 1
                result[1] = np.random.choice(filtered)

            # CD
            if cd.strip():
                lower, upper = self.cd_dict[cd]
                filtered = self.cd_samples[(self.cd_samples > lower) & (self.cd_samples <= upper)]
                attempts = 0
                while (len(filtered) < 20) and (attempts < 10):
                    lower = max(0, lower - 5)
                    upper = min(100, upper + 5)
                    filtered = self.cd_samples[(self.cd_samples > lower) & (self.cd_samples <= upper)]
                    attempts += 1
                result[2] = np.random.choice(filtered)

        except Exception as e:
            print(f"[ERROR] Index {idx}: {str(e)}\n{traceback.format_exc()}")
        return idx, result

    # ===== Run Sampling Function with threads =====
    def run_sampling(self, n_workers=4, result_dir=None):
        global result_array
        err_idx = []

        data_enum = list(enumerate(self.imsang_df[self.attribute_list].values))
        result_array = np.full((len(data_enum), len(self.attribute_list) + 2), np.nan)
        result_array[:, :2] = np.vstack((self.imsang_df.index, self.imsang_df['KOFTR_GROU'])).T
        

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, result in tqdm(executor.map(self.sample_attributes, data_enum), total=len(data_enum)):
                result_array[idx, 2:] = result
                if idx % 500 == 0:
                    header = 'ID,KOFTR_GROU' + ','.join(self.attribute_list)
                    output_path = os.path.join(result_dir, "HM변형1.0-SampledImsang.csv")
                    np.savetxt(
                        output_path,
                        result_array,
                        delimiter=',',
                        fmt='%s', 
                        header=header,
                        comments='' 
                    )
        return result_array, err_idx
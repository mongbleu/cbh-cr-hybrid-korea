"""SamplingImsang — class-conditional attribute sampling for the Imsang forest-type layer.

Each Imsang polygon/pixel carries *categorical* stand codes (crown density DNST_CD, diameter class
DMCLS_CD, height class HEIGHT). To feed the per-pixel CBH/CR model we need *continuous* values, so for
each row we draw a random sample from the empirical distribution of trees whose value falls inside that
code's [lower, upper] band, widening the band if too few samples are available.

Pipeline stage ⑤ (geospatial mapping). See .claude/skills/geospatial-raster-sampling and rules/geospatial.md.

Behavior is intentionally identical to the pre-2026-06-20 version (backup in src/_legacy/) — this pass
only reorganizes/annotates. Two pre-existing issues are flagged inline as `WARNING:` and reported
separately; they are NOT changed here because they affect map outputs and predate the WP-0 audit.
"""
import os                                            # filesystem path joins for the checkpoint CSV
import numpy as np                                   # array storage + random sampling
from tqdm.notebook import tqdm                       # progress bar inside Jupyter
import traceback                                     # full stack traces when a row fails
from concurrent.futures import ThreadPoolExecutor    # thread pool to sample many rows concurrently

# Default checkpoint filename written periodically during a run (kept as the historical name).
_OUTPUT_FILENAME = "HM변형1.0-SampledImsang.csv"      # module constant so the name lives in one place
_MIN_SAMPLES = 20                                    # minimum samples in a band before we stop widening
_MAX_ATTEMPTS = 10                                   # cap on widening iterations to avoid infinite loops


class SamplingImsang:                                # one instance configures + runs a full sampling pass
    def __init__(self, imsang_df, attribute_list=['DNST_CD', 'DMCLS_CD', 'HEIGHT'],
                 h_dict=None, dbh_dict=None, cd_dict=None,
                 h_samples=None, dbh_samples=None, cd_samples=None):
        self.imsang_df = imsang_df                   # source table: one row per Imsang unit, holds the codes
        self.attribute_list = attribute_list         # columns read per row, in order [DNST_CD, DMCLS_CD, HEIGHT]
        self.h_dict = h_dict                          # HEIGHT code -> (lower, upper) value band
        self.dbh_dict = dbh_dict                      # DMCLS_CD (diameter) code -> (lower, upper) band
        self.cd_dict = cd_dict                        # DNST_CD (crown density) code -> (lower, upper) band
        self.h_samples = h_samples                    # 1-D array of observed HEIGHT values to draw from
        self.dbh_samples = dbh_samples                # 1-D array of observed DBH values to draw from
        self.cd_samples = cd_samples                  # 1-D array of observed crown-density values to draw from

    @staticmethod
    def _sample_in_class(code, lookup, samples, step, lo_min, hi_max):
        """Draw one value from `samples` within the code's band, widening by `step` if too sparse.

        Returns np.nan when the code is blank (so the caller leaves the slot untouched). Identical to the
        original per-attribute blocks; only the shared logic is factored out here.
        """
        if not code.strip():                          # blank code -> nothing to sample for this attribute
            return np.nan                             # leave the result slot as NaN (matches original)
        lower, upper = lookup[code]                   # initial value band for this categorical code
        filtered = samples[(samples > lower) & (samples <= upper)]  # samples inside the (lower, upper] band
        attempts = 0                                  # count how many times we have widened the band
        while (len(filtered) < _MIN_SAMPLES) and (attempts < _MAX_ATTEMPTS):  # too few samples -> widen
            lower = max(lo_min, lower - step)         # push the lower edge down, clamped to lo_min
            upper = min(hi_max, upper + step)         # push the upper edge up, clamped to hi_max
            filtered = samples[(samples > lower) & (samples <= upper)]  # re-filter with the widened band
            attempts += 1                             # record the widening attempt
        return np.random.choice(filtered)             # pick one value uniformly at random from the band

    # ===== Sampling Function by Rows =====
    def sample_attributes(self, idx_data):            # worker: sample all attributes for a single row
        idx, data = idx_data                          # idx = row position; data = its [DNST_CD, DMCLS_CD, HEIGHT]
        result = [np.nan] * len(self.attribute_list)  # output slots, pre-filled with NaN
        try:                                          # any per-row failure is logged, not fatal to the run
            cd, dbh, h = data                         # unpack: cd=DNST_CD, dbh=DMCLS_CD, h=HEIGHT
            if (cd.strip() == '') or (dbh.strip() == '') or (h.strip() == ''):  # any code blank ->
                return idx, result                    # skip the row entirely, returning all-NaN
            # WARNING (pre-existing, not changed): result[0..2] are filled as H, DBH, CD, but the CSV
            # header (run_sampling) labels the columns in attribute_list order DNST_CD, DMCLS_CD, HEIGHT.
            # That reverses the H and CD labels. Verify against downstream consumers before trusting names.
            result[0] = self._sample_in_class(h, self.h_dict, self.h_samples,
                                               step=1, lo_min=0, hi_max=40)    # HEIGHT band widen ±1, [0,40]
            result[1] = self._sample_in_class(dbh, self.dbh_dict, self.dbh_samples,
                                               step=6, lo_min=0, hi_max=110)   # DBH band widen ±6, [0,110]
            result[2] = self._sample_in_class(cd, self.cd_dict, self.cd_samples,
                                               step=5, lo_min=0, hi_max=100)   # crown-density widen ±5, [0,100]
        except Exception as e:                        # e.g. empty band -> np.random.choice raises ValueError
            print(f"[ERROR] Index {idx}: {str(e)}\n{traceback.format_exc()}")  # log row + full traceback
        return idx, result                            # hand back position + sampled values to the collector

    # ===== Run Sampling Function with threads =====
    def run_sampling(self, n_workers=4, result_dir=None):  # drive the whole pass; periodically checkpoints
        err_idx = []                                  # reserved for failed indices (currently always empty)

        data_enum = list(enumerate(self.imsang_df[self.attribute_list].values))  # [(0,row0),(1,row1),...]
        result_array = np.full((len(data_enum), len(self.attribute_list) + 2), np.nan)  # cols: ID, group, +3
        result_array[:, :2] = np.vstack((self.imsang_df.index,             # col 0 = original row index
                                         self.imsang_df['KOFTR_GROU'])).T  # col 1 = KOFTR forest-group code
        # WARNING (pre-existing, not changed): header below omits a comma after KOFTR_GROU, fusing it with
        # the first attribute name (e.g. "KOFTR_GROUDNST_CD"). Fix needs the column-order check above first.
        header = 'ID,KOFTR_GROU' + ','.join(self.attribute_list)  # CSV header row (computed once)
        output_path = os.path.join(result_dir, _OUTPUT_FILENAME)  # full checkpoint path (computed once)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:  # spin up the worker threads
            # executor.map preserves input order, so results arrive idx = 0, 1, 2, ...
            for idx, result in tqdm(executor.map(self.sample_attributes, data_enum),  # sample rows in parallel
                                    total=len(data_enum)):       # total drives the tqdm progress bar
                result_array[idx, 2:] = result        # write this row's three sampled values into cols 2..4
                if idx % 500 == 0:                    # every 500 rows, checkpoint the full array to disk
                    np.savetxt(output_path,           # destination CSV path
                               result_array,          # the (partially filled) result matrix
                               delimiter=',',         # comma-separated
                               fmt='%s',              # write every cell as text (mixes ints/floats/NaN)
                               header=header,          # column header row
                               comments='')           # no leading '#' on the header line
        return result_array, err_idx                  # final matrix + (empty) failed-index list

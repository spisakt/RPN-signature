#!/usr/bin/env python
import pandas as pd
import PAINTeR.connectivity as conn
from PAINTeR import global_vars

# load bochum data
bochum_table = pd.read_csv(global_vars._RES_BOCHUM_TABLE_)

# load FD data
conn.add_FD_data(global_vars.bochum_fd_files, bochum_table)

# load timeseries data
conn.load_timeseries(global_vars.bochum_ts_files, bochum_table, scrubbing=True)

# compute connectivity

# define model

# train model with cross-validation

# estimate model accuracy with nested cross-validation

# serialise model

#save data.frame
bochum_table.to_csv(global_vars._RES_BOCHUM_TABLE_)
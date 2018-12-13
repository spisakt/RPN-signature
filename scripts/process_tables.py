#!/usr/bin/env python
import pandas as pd
import numpy as np
from PAINTeR import global_vars
from PAINTeR import misc
from PAINTeR import qst


# load tables
df_bochum = pd.read_csv(global_vars._BOCHUM_TABLE_)
df_essen = pd.read_csv(global_vars._ESSEN_TABLE_)
df_szeged = pd.read_csv(global_vars._SZEGED_TABLE_)


# calculate BMI
df_essen['BMI']=misc.bmi(df_essen['height'].values, df_essen['weight'].values)
df_szeged['BMI']=misc.bmi(df_szeged['height'].values, df_szeged['weight'].values)

# calculate mean QST thresholds
df_essen['CPT'] = qst.CPT( df_essen[['qst_cpt_2', 'qst_cpt_3', 'qst_cpt_4', 'qst_cpt_5', 'qst_cpt_6']].values,
                           truncate=True)
df_essen['HPT'] = qst.HPT( df_essen[['qst_hpt_2', 'qst_hpt_3', 'qst_hpt_4', 'qst_hpt_5', 'qst_hpt_6']].values)
df_essen['MPT_log_geom'] = qst.MPT( df_essen[['qst_mpt_1_pain', 'qst_mpt_1_no_pain',    # skip first?
                   'qst_mpt_2_pain', 'qst_mpt_2_no_pain',
                   'qst_mpt_3_pain', 'qst_mpt_3_no_pain',
                   'qst_mpt_4_pain', 'qst_mpt_4_no_pain',
                   'qst_mpt_5_pain', 'qst_mpt_5_no_pain']].values)

df_szeged['CPT'] = qst.CPT( df_szeged[['qst_cpt_2', 'qst_cpt_3', 'qst_cpt_4', 'qst_cpt_5', 'qst_cpt_6']].values,
                            truncate=True)
df_szeged['HPT'] = qst.HPT( df_szeged[['qst_hpt_2', 'qst_hpt_3', 'qst_hpt_4', 'qst_hpt_5', 'qst_hpt_6']].values)
df_szeged['MPT_log_geom'] = qst.MPT( df_szeged[['qst_mpt_1_pain', 'qst_mpt_1_no_pain',    # skip first?
                   'qst_mpt_2_pain', 'qst_mpt_2_no_pain',
                   'qst_mpt_3_pain', 'qst_mpt_3_no_pain',
                   'qst_mpt_4_pain', 'qst_mpt_4_no_pain',
                   'qst_mpt_5_pain', 'qst_mpt_5_no_pain']].values)


# calculate composite QST pain sensitivity (basedd on bochum sample mean and standard deviation)
print "Mean deviation from original pain sensitivity score:"
print np.mean(df_bochum['mean_QST_pain_sensitivity'] - qst.composite_pain_sensitivity(df_bochum['CPT'].values,
                                                                                      df_bochum['HPT'].values,
                                                                                      df_bochum['MPT_log_geom'].values))

df_essen['mean_QST_pain_sensitivity'] = qst.composite_pain_sensitivity_PAINTeR( df_essen['CPT'].values,
                                                                                df_essen['HPT'].values,
                                                                                df_essen['MPT_log_geom'].values,
                                                                                # reference:
                                                                                df_bochum['CPT'].values,
                                                                                df_bochum['HPT'].values,
                                                                                df_bochum['MPT_log_geom'].values)
df_szeged['mean_QST_pain_sensitivity'] = qst.composite_pain_sensitivity_PAINTeR(df_szeged['CPT'].values,
                                                                                df_szeged['HPT'].values,
                                                                                df_szeged['MPT_log_geom'].values,
                                                                                # reference:
                                                                                df_bochum['CPT'].values,
                                                                                df_bochum['HPT'].values,
                                                                                df_bochum['MPT_log_geom'].values)


# save data frames
#np.savetxt(global_vars._RES_BOCHUM_TABLE_, df_bochum.values)
df_bochum.to_csv(global_vars._RES_BOCHUM_TABLE_)
df_essen.to_csv(global_vars._RES_ESSEN_TABLE_)
df_szeged.to_csv(global_vars._RES_SZEGED_TABLE_)
from os import path

# atlas labels
_ATLAS_LABELS_ = path.join(path.dirname(__file__),"../data/atlas_relabeled.tsv")
_ATLAS_FILE_ = path.join(path.dirname(__file__),"../data/atlas_relabeled.nii.gz")

# input data.frames
_BOCHUM_TABLE_ = path.join(path.dirname(__file__),"../data/bochum-sample/bochum_sample.csv")
_ESSEN_TABLE_ = path.join(path.dirname(__file__),"../data/essen-sample/essen_sample.csv")
_SZEGED_TABLE_ = path.join(path.dirname(__file__),"../data/szeged-sample/szeged_sample.csv")


# input timeseries:
bochum_ts_files =[
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-004_pumi-0.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-005_pumi-1.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-006_pumi-2.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-007_pumi-3.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-008_pumi-4.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-009_pumi-5.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-010_pumi-6.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-011_pumi-7.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-012_pumi-8.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-013_pumi-9.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-014_pumi-10.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-015_pumi-11.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-016_pumi-12.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-017_pumi-13.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-018_pumi-14.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-019_pumi-15.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-020_pumi-16.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-023_pumi-17.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-024_pumi-18.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-025_pumi-19.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-026_pumi-20.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-027_pumi-21.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-028_pumi-22.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-029_pumi-23.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-030_pumi-24.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-031_pumi-25.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-032_pumi-26.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-033_pumi-27.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-034_pumi-28.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-035_pumi-29.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-036_pumi-30.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-037_pumi-31.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-038_pumi-32.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-039_pumi-33.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-040_pumi-34.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-041_pumi-35.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-042_pumi-36.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-043_pumi-37.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-044_pumi-38.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-045_pumi-39.tsv"),
path.join(path.dirname(__file__),"../data/bochum-sample/regional_timeseries/bochum-046_pumi-40.tsv")
]
essen_ts_files = [
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-001_pumi-44_pumi+045.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-002_pumi-43_pumi+044.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-003_pumi-42_pumi+043.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-004_pumi-41_pumi+042.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-005_pumi-18_pumi+019.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-006_pumi-39_pumi+040.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-007_pumi-38_pumi+039.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-008_pumi-37_pumi+038.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-009_pumi-36_pumi+037.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-010_pumi-35_pumi+036.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-011_pumi-33_pumi+034.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-012_pumi-34_pumi+035.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-013_pumi-32_pumi+033.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-014_pumi-31_pumi+032.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-015_pumi-30_pumi+031.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-016_pumi-28_pumi+029.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-017_pumi-29_pumi+030.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-018_pumi-27_pumi+028.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-019_pumi-26_pumi+027.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-020_pumi-25_pumi+026.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-021_pumi-24_pumi+025.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-022_pumi-23_pumi+024.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-023_pumi-14_pumi+015.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-024_pumi-22_pumi+023.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-025_pumi-21_pumi+022.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-026_pumi-20_pumi+021.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-027_pumi-19_pumi+020.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-028_pumi-17_pumi+018.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-029_pumi-16_pumi+017.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-030_pumi-13_pumi+014.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-031_pumi-15_pumi+016.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-032_pumi-11_pumi+012.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-033_pumi-12_pumi+013.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-034_pumi-10_pumi+011.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-035_pumi-9_pumi+010.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-036_pumi-8_pumi+009.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-037_pumi-7_pumi+008.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-038_pumi-6_pumi+007.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-039_pumi-5_pumi+006.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-040_pumi-2_pumi+003.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-041_pumi-1_pumi+002.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-042_pumi-4_pumi+005.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-043_pumi-0_pumi+001.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-044_pumi-3_pumi+004.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-045_pumi-l0_pumi+045.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-046_pumi-l1_pumi+046.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-047_pumi-l2_pumi+047.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-048_pumi-l3_pumi+048.tsv"),
path.join(path.dirname(__file__),"../data/essen-sample/regional_timeseries/essen-049_pumi-l4_pumi+049.tsv")
]
szeged_ts_files = [
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-001_pumi-0.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-002_pumi-1.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-003_pumi-2.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-004_pumi-3.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-005_pumi-4.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-006_pumi-5.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-007_pumi-6.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-008_pumi-7.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-009_pumi-8.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-010_pumi-9.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-011_pumi-10.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-012_pumi-11.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-013_pumi-12.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-014_pumi-13.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-015_pumi-14.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-016_pumi-15.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-017_pumi-16.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-018_pumi-17.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-019_pumi-18.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-020_pumi-19.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-021_pumi-20.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-022_pumi-21.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-023_pumi-22.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-024_pumi-23.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-025_pumi-24.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-026_pumi-25.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-027_pumi-26.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-028_pumi-27.tsv"),
path.join(path.dirname(__file__),"../data/szeged-sample/regional_timeseries/ts_szeged-029_pumi-28.tsv")
]


# inoput FD files
bochum_fd_files = [
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-004_pumi-0.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-005_pumi-1.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-006_pumi-2.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-007_pumi-3.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-008_pumi-4.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-009_pumi-5.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-010_pumi-6.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-011_pumi-7.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-012_pumi-8.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-013_pumi-9.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-014_pumi-10.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-015_pumi-11.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-016_pumi-12.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-017_pumi-13.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-018_pumi-14.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-019_pumi-15.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-020_pumi-16.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-023_pumi-17.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-024_pumi-18.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-025_pumi-19.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-026_pumi-20.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-027_pumi-21.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-028_pumi-22.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-029_pumi-23.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-030_pumi-24.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-031_pumi-25.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-032_pumi-26.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-033_pumi-27.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-034_pumi-28.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-035_pumi-29.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-036_pumi-30.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-037_pumi-31.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-038_pumi-32.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-039_pumi-33.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-040_pumi-34.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-041_pumi-35.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-042_pumi-36.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-043_pumi-37.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-044_pumi-38.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-045_pumi-39.txt"),
path.join(path.dirname(__file__),"../data/bochum-sample/framewise_displacement/FD_bochum-046_pumi-40.txt")
]
essen_fd_files = [
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-001_pumi-44_pumi+045.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-002_pumi-43_pumi+044.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-003_pumi-42_pumi+043.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-004_pumi-41_pumi+042.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-005_pumi-18_pumi+019.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-006_pumi-39_pumi+040.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-007_pumi-38_pumi+039.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-008_pumi-37_pumi+038.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-009_pumi-36_pumi+037.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-010_pumi-35_pumi+036.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-011_pumi-33_pumi+034.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-012_pumi-34_pumi+035.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-013_pumi-32_pumi+033.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-014_pumi-31_pumi+032.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-015_pumi-30_pumi+031.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-016_pumi-28_pumi+029.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-017_pumi-29_pumi+030.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-018_pumi-27_pumi+028.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-019_pumi-26_pumi+027.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-020_pumi-25_pumi+026.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-021_pumi-24_pumi+025.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-022_pumi-23_pumi+024.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-023_pumi-14_pumi+015.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-024_pumi-22_pumi+023.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-025_pumi-21_pumi+022.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-026_pumi-20_pumi+021.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-027_pumi-19_pumi+020.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-028_pumi-17_pumi+018.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-029_pumi-16_pumi+017.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-030_pumi-13_pumi+014.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-031_pumi-15_pumi+016.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-032_pumi-11_pumi+012.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-033_pumi-12_pumi+013.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-034_pumi-10_pumi+011.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-035_pumi-9_pumi+010.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-036_pumi-8_pumi+009.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-037_pumi-7_pumi+008.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-038_pumi-6_pumi+007.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-039_pumi-5_pumi+006.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-040_pumi-2_pumi+003.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-041_pumi-1_pumi+002.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-042_pumi-4_pumi+005.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-043_pumi-0_pumi+001.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-044_pumi-3_pumi+004.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-045_pumi-l0_pumi+045.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-046_pumi-l1_pumi+046.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-047_pumi-l2_pumi+047.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-048_pumi-l3_pumi+048.txt"),
path.join(path.dirname(__file__),"../data/essen-sample/framewise_displacement/FD_essen-049_pumi-l4_pumi+049.txt")
]
szeged_fd_files = [
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-001_pumi-0.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-002_pumi-1.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-003_pumi-2.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-004_pumi-3.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-005_pumi-4.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-006_pumi-5.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-007_pumi-6.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-008_pumi-7.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-009_pumi-8.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-010_pumi-9.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-011_pumi-10.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-012_pumi-11.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-013_pumi-12.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-014_pumi-13.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-015_pumi-14.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-016_pumi-15.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-017_pumi-16.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-018_pumi-17.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-019_pumi-18.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-020_pumi-19.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-021_pumi-20.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-022_pumi-21.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-023_pumi-22.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-024_pumi-23.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-025_pumi-24.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-026_pumi-25.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-027_pumi-26.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-028_pumi-27.txt"),
path.join(path.dirname(__file__),"../data/szeged-sample/framewise_displacement/FD_szeged-029_pumi-28.txt")
]


# output data.frames
_RES_BOCHUM_TABLE_ = path.join(path.dirname(__file__),"../res/bochum_sample.csv")
_RES_ESSEN_TABLE_ = path.join(path.dirname(__file__),"../res/essen_sample.csv")
_RES_SZEGED_TABLE_ = path.join(path.dirname(__file__),"../res/szeged_sample.csv")

_RES_BOCHUM_TABLE_EXCL_ = path.join(path.dirname(__file__),"../res/bochum_sample_excl.csv")
_RES_ESSEN_TABLE_EXCL_ = path.join(path.dirname(__file__),"../res/essen_sample_excl.csv")
_RES_SZEGED_TABLE_EXCL_ = path.join(path.dirname(__file__),"../res/szeged_sample_excl.csv")

# output features
_FEATURE_BOCHUM_ = path.join(path.dirname(__file__),"../res/feature_bochum.sav")
_FEATURE_ESSEN_ = path.join(path.dirname(__file__),"../res/feature_essen.sav")
_FEATURE_SZEGED_ = path.join(path.dirname(__file__),"../res/feature_szeged.sav")

# output predictive model
_RES_PRED_MOD_ = path.join(path.dirname(__file__),"../res/predictive_model_temp.sav")
_RES_PRED_MOD_FIXED_ = path.join(path.dirname(__file__),"../res/predictive_model.sav")

_RES_PRED_CONN_ = path.join(path.dirname(__file__),"../res/predictive_connections.csv")

# output plots:
_PLOT_BOCHUM_MEAN_MATRIX_ = path.join(path.dirname(__file__),"../res/mtx_mean_bochum.pdf")
_PLOT_ESSEN_MEAN_MATRIX_ = path.join(path.dirname(__file__),"../res//mtx_mean_essen.pdf")
_PLOT_SZEGED_MEAN_MATRIX_ = path.join(path.dirname(__file__),"../res//mtx_mean_szeged.pdf")
_PLOT_PRED_MATRIX_ = path.join(path.dirname(__file__),"../res//mtx_pred.pdf")

_PLOT_BOCHUM_PREDICTION_ = path.join(path.dirname(__file__),"../res/pred_bochum.pdf")
_PLOT_ESSEN_PREDICTION_ = path.join(path.dirname(__file__),"../res/pred_essen.pdf")
_PLOT_SZEGED_PREDICTION_ = path.join(path.dirname(__file__),"../res/pred_szeged.pdf")
_PLOT_ESSEN_SZEGED_PREDICTION_ = path.join(path.dirname(__file__),"../res/pred_essen_szeged.pdf")

_PLOT_LEARNING_CURVE_ = path.join(path.dirname(__file__),"../res/learning_curve.pdf")
<img src="https://github.com/tscnlab/Templates/blob/main/logo/logo_with_text-01.png" width="400"/>

# Overview

Repository for the photic sneezing case study report: **Sneezing in response to bright light exposure: A case study in a photic sneezer**, publicly accessible under the [MIT](https://opensource.org/license/mit/) license (see `LICENSE.md` file).

# Reproducing the analysis

## Cloning the repository

This repository first needs to be cloned to a local directory on your machine.

## Dependency management

This Python project supports [uv](https://docs.astral.sh/uv/) and [pixi](https://pixi.sh/latest/) for package management, using a [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) configuration file.

If you use either: open the terminal, cd to the directory of the cloned repository, and type `uv sync` or `pixi install --locked` (or other) to set up a virtual environment with the correct packages installed and with the correct versions.

## Downloading the data

1. The data must be downloaded from the Max Planck Digital Library (MDPL) data repository [Edmond](https://doi.org/10.17617/3.LO8EXZ). You can choose between downloading the exports only (around 4 Gb) or the full dataset including eye videos (200 Gb).
2. The downloaded data must then placed in the cloned repository directory. **The folder should be renamed to `data`!**

### Data structure

The data contains:

1. Data for both the "30-min" and "one-shot" laboratory experimental parts: `data/lab_data/30_min_exp` and `data/lab_data/one_shot_exp`.
    Each trial folder contains both:
   
    a. Pupillary data, including eye video recordings and Pupil Capture exports (needed for pupil size analysis)
   
    b. Subjective feeling reports, under the format `sneeze_data_[PID].csv`.
   
    *NOTES: Trials 25 and 53 do not have sneeze data files.*

3. Real-world light exposure and sneeze log data: `data/real_world_data`.
    The folder contains:
   
    a. ActTrust2 light exposure data `actiwatch_combined_data.txt`, this file is the combined version of the three raw files `Log_[…].txt`
   
    b. Sneeze logging data `date_times_sneeze_log_notion.csv`, this file contains self-recorded timestamps of sneeze-related events

#### Directory tree

*not included in `data_exports_only/`

```text
/root
├── data_full/                                      # original unprocessed data + processed data
    ├── lab_data/                                   ## data for the laboratory part of the experiment
        ├── 30_min_exp/                             ### data for the 30-min paradigm
            └── XXX                                 #### 3 individual recording folders
        └── one_shot_exp/                           ### data for the one-shot paradigm
            └── XXX                                 #### 63 individual recording folders
    └── real_world_data/                            ## data for the real-world part of the experiment
        ├── actiwatch_combined_data.txt             ### processed light exposure data
        ├── Log_1667_2022[…].txt                *   ### raw light exposure data
        └── date_times_sneeze_log_notion.csv        ### timestamps of sneeze events
└── data_exports_only/                              # processed data only
    └── …                                           ## identical structure to data_full/
```

The structure of the individual trial folders is as follows, generated from the Pupil Core software. More info:
- <https://docs.pupil-labs.com/core/developer/recording-format/>
- <https://docs.pupil-labs.com/core/software/recording-format/>
- <https://docs.pupil-labs.com/core/software/pupil-player/#export>

The subjective reporting data was not generated using Pupil Core but added there for convenience.

```text
/XXX
├── exports/
    └── 000
        ├── annotations.csv                         # experimenter annotations
        ├── export_info.csv                         # export metadata
        ├── pupil_gaze_positions_info.txt
        ├── pupil_positions.csv                     # contains pupil position and diameter data
        ├── world_timestamps.csv
        └── world_timestamps.npy
├── annotation_player_timestamps.npy            *
├── annotation_player.pldata                    *
├── annotation_timestamps.npy                   *
├── annotation.pldata                           *
├── blinks_timestamps.npy                       *
├── blinks.pldata                               *
├── eye1_lookup.npy                             *
├── eye1_timestamps.npy                         *
├── eye1.intrinsics                             *
├── eye1.mp4                                    *
├── fixations_timestamps.npy                    *
├── fixations.pldata                            *
├── gaze_timestamps.npy                         *
├── gaze.pldata                                 *
├── info.player.json                                # recording metadata
├── notify_timestamps.npy                       *
├── notify.pldata                               *
├── pupil_timestamps.npy                        *
├── pupil.pldata                                *
├── user_info.csv                               *
├── world_lookup.npy                            *
├── world_timestamps.npy                        *
├── world.intrinsics                            *
├── world.mp4                                   *
└── sneeze_data_[…].csv                             # subjective reporting data
```

The terminal command used to skim down the full dataset is the following:

```bash
# delete unwanted files
find . -type f \( \
    \( ! -name 'sneeze_*' \
       ! -name 'info*' \
       ! -name 'actiwatch_*' \
       ! -name 'date_times_*' \
       ! -path '*/exports/*' \) \
    -o -name '*.mp4' \
\) -exec rm -f {} +

# remove empty folders (but keep any folder literally named "exports")
find . -type d -empty -not -name 'exports' -delete
```

## Running the analysis script

The `analysis.ipynb` file contains all the code for running the analysis of the data and generating the final figure.

- Input: data in `data` folder, downloaded and copied as described above

- Output: PDF figure, saved at `outputs/plot.pdf`

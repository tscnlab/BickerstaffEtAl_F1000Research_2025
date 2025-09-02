<img src="https://github.com/tscnlab/Templates/blob/main/logo/logo_with_text-01.png" width="400"/>

# Overview

Repository for the photic sneezing case study report: **Sneezing in response to bright light exposure: A case study in a photic sneezer**, publicly accessible under the [MIT](https://opensource.org/license/mit/) license (see `LICENSE.md` file).

# Reproducing the analysis

## Cloning the repository

This repository first needs to be cloned to a local directory on your machine.

## Dependency management

This Python project supports [uv](https://docs.astral.sh/uv/) and [pixi](https://pixi.sh/latest/) for package management, using a [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) configuration file.
Open the terminal, cd to the directory of the cloned repository, and type `uv sync` to set up a virtual environment with the correct packages installed and with the correct versions.

## Downloading data

1. The data must be downloaded from an online repository ([…] Gb), accessible at [URL]
2. The downloaded data must then placed in the cloned repository. The folder should be named `data`.

### Data structure

The data contains:

1. Data for both the "30-min" and "one-shot" laboratory experimental parts: `data/lab_data/30_min_exp` and `data/lab_data/one_shot_exp`
    Each trial folder contains both:
    a. Pupillary data, including eye video recordings and Pupil Capture exports (needed for pupil size analysis)
    b. Subjective feeling reports, under the format `sneeze_data_[PID].csv`.
    NOTES: Trials 25 and 53 do not have sneeze data files.

2. Real-world light exposure and sneeze log data: `data/real_world_data`
    The folder contains:
    a. ActTrust2 light exposure data `actiwatch_combined_data.txt`, this file is the combined version of the three raw files `Log_[…].txt`
    b. Sneeze logging data `date_times_sneeze_log_notion.csv`, this file contains self-recorded timestamps of sneeze-related events

## Running the analysis script

The `analysis.ipynb` file contains all the code for running the analysis of the data and generating the final figure.

Input: data in `data` folder, downloaded and copied as described above
Output: PDF figure, saved at `outputs/plot.pdf`

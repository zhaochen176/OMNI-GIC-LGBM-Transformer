# Data Description and Availability

This directory describes the data sources used in this study and provides guidance on data access and usage for reproducibility purposes.

## OMNI Solar Wind Data

The upstream solar wind and interplanetary magnetic field (IMF) parameters used in this study are obtained from the **OMNI database** maintained by NASA’s Space Physics Data Facility.

- Data portal: https://omniweb.gsfc.nasa.gov/
- Time resolution: 1 minute
- Parameters: solar wind plasma parameters, IMF components, and geomagnetic indices

The OMNI data are **publicly available** and can be freely accessed from the OMNIWeb interface.  
Users can reproduce the solar wind input data used in this study by selecting the same parameters, time intervals, and time resolution as specified in the manuscript and configuration files.

## Public GIC Measurement Dataset

To demonstrate the general applicability of the proposed GIC prediction framework and to facilitate reproducibility, this repository supports the use of a **publicly available GIC measurement dataset** archived on Zenodo:

- Dataset DOI: https://doi.org/10.5281/zenodo.3714269

This dataset provides measured geomagnetically induced current (GIC) time series and can be directly used with the preprocessing and modeling pipeline implemented in this repository.

## GIC Measurements Used in the Manuscript

In the accompanying manuscript, part of the analysis is based on **measured GIC data from the Ling’ao Nuclear Power Station (China)**.

Due to data-sharing agreements and operational confidentiality requirements, the original GIC measurements from this site **cannot be publicly released**. As a result, these specific raw data are not redistributed in this repository.

Nevertheless, the following measures are taken to ensure scientific transparency and reproducibility:

- All data preprocessing procedures, feature construction methods, and model architectures are fully documented.
- The complete modeling pipeline is provided and can be executed using publicly available GIC datasets.
- The public Zenodo dataset serves as an open reference example to validate the methodology and reproduce the workflow.
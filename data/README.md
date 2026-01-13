### Data Sources
1. **OMNI solar wind data**: obtained from NASA/GSFC Space Physics Data Facility (OMNI/OMNIWeb).
2. **GIC measurements**: ground-based observations from the GIC monitoring system used in this study.

### Preprocessing Summary 
- Time alignment between OMNI parameters and GIC observations (1-minute resolution).
- Data cleaning (invalid values and outliers handling).
- Missing-value treatment (interpolation for short gaps).
- Construction of physics-informed features (e.g., Akasofu ε) and geomagnetic indices derivatives (e.g., SYM-H derivatives).
- Solar wind feature lagging (shifting upstream parameters by a fixed lead time) to account for propagation/response delay.
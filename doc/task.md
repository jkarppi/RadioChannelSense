# Project Assignment Guidelines

The project applies course techniques to practical wireless applications.

## 1. Dataset Creation

Create a spatially consistent dataset. You may consider:

- Single or multiple base stations
- Single or multiple frequencies
- Indoor or outdoor scenarios

You can use any simulator or public dataset, for example:

- Quadriga
- MATLAB Ray Tracing
- RemCom Ray Tracing
- Sionna RT
- DeepMIMO
- University of Stuttgart DICHASUS
- Fraunhofer UWB and 5G dataset
- CAEZ (ETH Zurich CSI Acquisition)

For UE location sampling, choose a suitable sample density. A few samples per $m^2$ is typically sufficient for many radio resource management applications and can support localization accuracy around $1\,m$.

### Antenna guidance

- BS antenna: 8-32 elements is sufficient
- Consider sectorized arrays and realistic antenna patterns to improve channel distinguishability
- UE antenna:
  - Sub-6 GHz: single antenna is acceptable
  - mmWave: multi-antenna at both UE and BS (with beamforming)
  - Around 8 UE antennas is sufficient

## 2. Fingerprint Localization

Use the dataset for fingerprint localization with both:

- WKNN
- NN-based localization

You are free to choose:

- CSI features
- Distance metric
- Weight function
- NN architecture

Evaluate performance with:

- RMSE
- Error CDF
- Heatmaps of localization error

You may use both classification and regression approaches.

## 3. Channel Charting

Create a channel chart for the radio environment using both:

- Conventional DR methods
- NN-based methods

Evaluate using:

- TW
- CT
- KS
- Chart visualizations

When selecting DR parameters, show how they affect performance using a suitable cost function or metric.

At minimum, present channel charting results for two techniques:

- One method covered in class
- One additional method (for example, UMAP)

For resources, see: **What is Channel Charting? | Channel Charting Resources**

## 4. Extra Credit Options

You can get extra points by including one or more of the following:

1. Use a customized dataset (for example Aalto campus, Helsinki center, your city) with MATLAB RT, Sionna RT, or RemCom RT
2. In Quadriga, use site-specific simulations (for example Madrid layout)
3. In Quadriga, use spherical wavefront and a near-field scenario
4. Consider multi-point channel charting
5. Use channel charting for an application (for example beam management)
6. Address out-of-sample generalization for channel charting
7. Consider estimated channels with noise and impairments
8. Consider semi-supervised channel charting

## 5. If Channel Charting Quality Is Poor

If you cannot obtain a good channel chart from your own dataset:

- First analyze and discuss the dataset limitations
- Then you may use DeepMIMO or measured datasets (for example ETHZ, DICHASUS)

Additional useful datasets:

- MATLAB localization dataset: *Three-Dimensional Indoor Positioning with 802.11az Fingerprinting and Deep Learning*
- Sionna RT dataset: https://zenodo.org/records/14535165

## 6. Technical Support

- Channel charting: Pere (`pere.garauburguera@aalto.fi`)
- Fingerprint localization: Xinze (`xinze.li@aalto.fi`)
- Neural networks: Ashvin (`ashvin.1.srinivasan@aalto.fi`)
- You can also contact the course instructor

## Report Requirements

1. Write a **2-5 page** report covering: dataset, fingerprint localization, and channel charting
2. Include figures for UE locations, sample CSI, and summary tables
3. Show channel charting and fingerprint localization results
4. Discuss challenging issues
5. Use proper references and cite all reused figures/block diagrams

## Dataset and Code Submission

Provide a downloadable link to the dataset and selected code.

## Presentation Requirements

- Group presentation length: **10-12 minutes**
- Each group member should present part of the results
- Use clear, representative slides
- You may discuss NN structures, distance metrics, CSI features, DR methods, etc.

Attach presentation slides to the final project submission.

**Deadline:** 10:00 AM, 17 April 2026

A submission link will be provided for report, slides, and additional materials.

All group members are expected to share responsibility for completing the project.
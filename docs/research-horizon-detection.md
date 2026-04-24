# Real-Time Horizon Detection for UAVs – Literature Review, Datasets and Test Design (April 2026)

## 1 Problem context and hackathon requirements

The **DroneAidCollective** hackathon problem asks participants to implement a real-time horizon detection algorithm for a camera-equipped UAV that will run simultaneously with a Hailo accelerator on a Raspberry Pi 5. The goal is to crop out the sky portion of each frame so that a downstream ground-target detector does not waste resources processing sky pixels and is less susceptible to false positives (e.g., clouds, birds). The evaluation criteria require:

- **Output:** horizon line parameters (slope/angle and vertical offset) or a binary sky mask.
- **Performance:** ≥ 15 frames/s (FPS) on a Raspberry Pi 5 while the detector runs concurrently on the Hailo accelerator.
- **Accuracy:** correct estimation of the horizon line's angle and offset.

The constraints imply that algorithms must be light-weight (ideally classical computer vision) and robust to varied environments (different altitudes, lighting and weather).

## 2 Existing solutions and literature

Horizon detection has been studied for ship navigation, UAV attitude estimation and geo-localisation for decades. The following summarises key approaches that are relevant to a non-learning implementation:

### 2.1 Thresholding and morphological approaches

- **Otsu thresholding:** A simple method implemented in the GitHub `sallamander/horizon-detection` project. The image is blurred, thresholded using Otsu's method and morphologically closed; the horizon is taken as the last row with non-sky pixels at the left and right image boundaries ([ref 1](https://raw.githubusercontent.com/sallamander/horizon-detection/master/utils.py)). The method is extremely fast, but its accuracy deteriorates in complex scenes (e.g., buildings and hills). Similar binary-threshold approaches have been used to create sky masks.

- **Hough-transform variants:** Edge-based techniques detect strong edge segments and fit a line via the Hough transform. A fast maritime detector proposes extracting line segments at a very low edge threshold, filtering segments by length and slope, grouping collinear segments and selecting the group forming the longest line; vectorised computations allow real-time operation on CPU ([ref 2](https://arxiv.org/html/2110.13694v4)). These methods work well when the horizon is roughly straight but fail when buildings, hills or waves create non-linear boundaries.

- **Morphological gradient search:** Some algorithms compute vertical gradients and search for the highest gradient peak along each column to approximate the horizon. Median filtering and morphological closing remove noise, and a line is fitted through the selected points. Although simple, gradient methods can be confused by strong gradients from roads or roofs.

### 2.2 Statistical modelling of sky/ground

- **Ettinger's covariance-based method:** Designed for micro-air vehicles, this algorithm models the sky and ground regions as Gaussian colour distributions. For each candidate line, it computes the means and covariance matrices of the pixels on both sides of the line and maximises a cost function based on the determinants and eigenvalues of the covariances to find the best separator ([ref 3](https://github.com/k29/horizon_detection/blob/master/myHorizon.py)). The model is updated over time using running means and covariances, and a detector of extreme attitudes checks when the horizon is out of view ([ref 4](https://www.eecis.udel.edu/~cer/arv/readings/paper_ettinger.pdf)). Because it relies only on simple arithmetic and eigenvalue calculations, it can be implemented efficiently in C++/NumPy. The method assumes the sky–ground boundary is roughly straight; it is sensitive to colour similarity between sky and ground (e.g., grey skies).

- **Fuzzy logic and region growing:** Some real-time detectors operate in YUV colour space. They define fuzzy subsets for sky and ground based on Y, U and V values and use morphological operations to refine the mask. These methods require manual tuning but can be robust to lighting changes ([ref 5](https://www.mdpi.com/2079-9292/9/4/614)).

### 2.3 Vanishing-point and geometric methods

- **Vanishing-point-based detectors:** A research project at Eötvös Loránd University (2025) explores estimating the horizon in urban scenes via vanishing points. The slope and offset of the horizon are estimated from the positions of horizontal vanishing points; preliminary results show real-time performance ([ref 6](https://cv.inf.elte.hu/index.php/2025/12/19/ongoing-project-horizon-line-detection/)). Vanishing-point methods exploit perspective geometry and handle scenes with dominant linear structures (buildings, roads).

- **Horizon/vanishing-point estimation via classification:** A 2021 presentation treats the horizon line and horizontal vanishing points as classification problems. The horizon is parameterised by its slope angle and vertical offset, and a convolutional neural network (CNN) predicts probability distributions over these parameters ([ref 7](https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Elassam.pdf)). Although this method is deep-learning based, the parameterisation (angle, offset) matches the hackathon's evaluation metric.

### 2.4 Other notable methods

- **Temporal and optical-flow methods:** For video, combining optical flow with the Hough transform helps remove transient edges caused by moving objects.

- **Hidden-Markov tree segmentation and SVM/clustering:** These build pixel classifiers (e.g., support-vector machines or Gaussian mixture models) for sky and ground and use Markov trees to enforce spatial consistency ([ref 5](https://www.mdpi.com/2079-9292/9/4/614)). They are more computationally intensive and thus less suited to a Raspberry Pi without hardware acceleration.

- **Deep-learning segmentation:** Modern works use U-Net or encoder–decoder networks to segment sky and ground. For example, the `horizon-uav` repository performs semantic segmentation using a U-Net variant, extracts the border between sky and ground using morphology, fits a line by linear regression and converts it to pitch and roll angles ([ref 8](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)). The pre-trained model runs in real-time on a Raspberry Pi 4 with a Coral TPU. Deep segmentation methods can handle curved horizons but may not meet the no-learning requirement unless a very light model and accelerator are available.

### 2.5 Surveys and comparisons

- A comparison of semantic segmentation approaches for horizon detection emphasises that purely linear models struggle in complex terrain (mountains, buildings); deep segmentation networks (FCN, SegNet) improve robustness but require more computation.
- A review of horizon detection algorithms summarises methods such as statistical models, hidden Markov trees, SVMs, gradient-based detectors, optical-flow with Hough transforms and fuzzy classification ([ref 5](https://www.mdpi.com/2079-9292/9/4/614)). Understanding their assumptions and computational complexity helps in selecting an algorithm.

## 3 Datasets and sources of drone footage

A test dataset is necessary both for tuning algorithm parameters and for benchmarking. For the hackathon, a mix of labelled and unlabelled datasets can be used.

### 3.1 Publicly available horizon-line datasets

| Dataset (year) | Description | Pros/cons |
|---|---|---|
| **KITTI Horizon dataset (2020)** | Created as part of the ["Temporally Consistent Horizon Lines"](https://ris.utwente.nl/ws/files/254342269/yang_tem.pdf) paper. It contains **43 699 images across 72 video sequences** from the KITTI autonomous-driving dataset with manually annotated horizon line parameters ([ref 9](https://ris.utwente.nl/ws/files/254342269/yang_tem.pdf)). Suitable for training/testing line-based detectors; includes diverse urban and rural scenes. | Horizon annotations follow the slope/offset parameterisation; dataset can be used for evaluating accuracy and speed. Not drone specific (ground-vehicle perspective) but still useful. |
| **Fast maritime horizon dataset (2020)** | Provided by the paper ["A fast horizon detector and a new annotated dataset for maritime video processing."](https://arxiv.org/html/2110.13694v4) Includes maritime videos with annotated horizon lines; algorithm uses low-threshold line segments ([ref 2](https://arxiv.org/html/2110.13694v4)). | Useful for sea-sky boundaries (similar to high-altitude drone views). Scenes are water-dominant; limited diversity for land operations. |
| **Horizon-UAV dataset (2021)** | The [horizon-uav](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md) GitHub repository offers an annotated dataset (downloadable via script or OneDrive) for UAV horizon detection. The dataset contains images and videos captured from UAVs with labelled sky–ground masks. The repository uses this dataset to train a U-Net and notes that the dataset can be downloaded via `download_dataset_original.py` or from OneDrive ([ref 10](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)). | Contains front-facing UAV footage with annotated sky masks. Ideal for training segmentation or testing threshold/morphology methods. |
| **Sky segmentation datasets – SkyFinder and SUN-Sky (2017), [Sky Segmentation Dataset](https://maadaa.ai/datasets/DatasetsDetail/Sky-Segmentation-Dataset) (~73.6k images)** | Provide images from webcams and general scenes with sky masks. Horizon lines can be derived by taking the upper contour of the ground mask ([ref 11](https://maadaa.ai/datasets/DatasetsDetail/Sky-Segmentation-Dataset)). | Very large and diverse; annotation quality varies; masks include curved horizons (mountains, buildings). |
| **Other datasets (VisDrone, Kaggle drone videos, MCL dataset, FRED)** | Large drone-footage datasets aimed at object detection. They lack horizon annotations but supply varied aerial scenes that can be manually labelled. Kaggle's "Drone Videos" dataset provides raw videos; VisDrone contains over 10k images and 263 videos; FRED combines RGB and event data. | Good for creating a custom test set. Annotation is required. |

### 3.2 Generating a custom test dataset

If existing datasets are insufficient, you can collect your own video using a drone and annotate the horizon using tools such as **LabelMe**, **CVAT** or **LabelBox**. Annotation involves marking two points on the horizon for each frame to compute the slope (angle) and intercept. To create a sky mask, you can annotate polygons for the sky region. The [horizon-uav](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md) repository uses LabelBox and provides a script to convert the JSON labels into training masks ([ref 12](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)). Including varied altitudes, camera pitches and weather conditions will improve robustness.

## 4 Designing tests to verify the algorithm

A rigorous evaluation should measure both **accuracy** and **performance**.

### 4.1 Metrics

- **Slope/angle error:** Represent the horizon line by `y = m x + b`, where `m = tan(θ)` and `θ` is the angle relative to the x-axis. Compute the absolute angular error `|θ_pred - θ_gt|` in degrees.
- **Offset error:** Use the vertical intercept `b` (in pixels). Compute the absolute difference `|b_pred - b_gt|` or normalise by image height.
- **Mask overlap:** If a sky mask is predicted, calculate Intersection-over-Union (IoU) between the predicted and ground-truth masks.
- **Processing speed:** Measure the average FPS on the Raspberry Pi 5 while concurrently running the downstream detector on the Hailo accelerator. Consider measuring CPU utilisation and memory footprint.

The 2021 horizon-and-vanishing-point work parameterised the horizon by its slope angle and offset, which aligns with the above metrics ([ref 7](https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Elassam.pdf)).

### 4.2 Test procedure

1. **Prepare a labelled dataset.** Use a mix of KITTI Horizon, Horizon-UAV, and custom annotated frames that represent your target use case (drone altitudes, camera tilts). Split into training (if tuning parameters) and testing sets.
2. **Implement the algorithm.** For a classical CV approach, start with a simple baseline (e.g., Otsu threshold and morphological closing) and gradually refine. Parameter tuning can use a grid search on the training set (kernel sizes, threshold levels, gradient filters). If using statistical modelling, implement Ettinger's covariance-based method and select the cost function weight. Vectorise operations to maximise speed on ARM CPUs.
3. **Run inference on the test set.** For each frame, record predicted line parameters, compute the errors relative to ground truth and aggregate statistics (mean absolute error, standard deviation, percentage of frames within a tolerance). For mask outputs, compute IoU.
4. **Evaluate speed.** Deploy the algorithm to a Raspberry Pi 5 and measure FPS with and without the downstream detector. Use synthetic workloads (e.g., dummy Hailo process) if the detector is not yet available.
5. **Analyse failure cases.** Identify frames where the error is large (e.g., tall buildings occlude horizon, grey skies reduce contrast). Use this insight to improve the algorithm (e.g., include line-segment grouping or incorporate inertial sensor cues).

## 5 Building the test set first and then using AI to create a solution

One strategy is to **construct the test dataset before developing the algorithm** and then let AI search for a solution that performs well on this dataset. This can be done in two ways:

1. **Parameter optimisation for classical CV algorithms:** Define a parameterised pipeline—e.g., Gaussian blur size, Otsu threshold or manual threshold value, gradient filter size, morphological kernel dimensions, line-fitting method (least squares, RANSAC). Write an evaluation script that returns the accuracy and processing time on the test set. Use a brute-force or Bayesian optimisation search over the parameter space to find the configuration that minimises the error while satisfying the FPS constraint. Because the parameter space is small (perhaps 5–10 parameters), exhaustive search is feasible on a workstation. This approach allows you to treat the pipeline as a black box and automatically tune it.

2. **AutoML for lightweight models:** If you decide that a minimal neural network is acceptable (e.g., a tiny U-Net on the Hailo accelerator), you can design a search space over network architectures (number of layers, filter sizes, quantisation). Use a neural architecture search library to train candidate models on your dataset, evaluate them on accuracy and inference speed on the target hardware, and select the best model. The [horizon-uav](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md) repository supports training and quantisation for Raspberry Pi + Coral TPU ([ref 13](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)). However, this requires more development time and may conflict with the hackathon's "classical algorithm" suggestion.

Regardless of the approach, the key is to build a representative dataset that covers the expected operating conditions. Without such diversity, an algorithm (whether hand-crafted or learned) will overfit and fail in the field.

## 6 Summary and recommendations

- **Existing solutions** range from simple thresholding and line detection to statistical modelling and deep segmentation. **Ettinger's covariance-based method** offers an efficient, classical approach suitable for an embedded CPU ([ref 3](https://github.com/k29/horizon_detection/blob/master/myHorizon.py)). **Otsu thresholding with morphology** is extremely fast but fails in complex scenes ([ref 1](https://raw.githubusercontent.com/sallamander/horizon-detection/master/utils.py)). **Vanishing-point methods** provide robustness in urban environments ([ref 6](https://cv.inf.elte.hu/index.php/2025/12/19/ongoing-project-horizon-line-detection/)), and **U-Net segmentation followed by line fitting** achieves high accuracy but needs hardware acceleration ([ref 8](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)).

- **Datasets** for horizon detection include the **KITTI Horizon dataset** (43 699 labelled images) ([ref 9](https://ris.utwente.nl/ws/files/254342269/yang_tem.pdf)), the **Horizon-UAV dataset** with sky masks and videos ([ref 10](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)), maritime datasets ([ref 2](https://arxiv.org/html/2110.13694v4)), and large sky segmentation datasets ([ref 11](https://maadaa.ai/datasets/DatasetsDetail/Sky-Segmentation-Dataset)). Unlabelled drone footage (e.g., Kaggle, VisDrone) can be annotated to create a custom test set.

- **Test design** should measure slope/offset errors and processing speed, and include diverse scenes. The horizon line can be parameterised by angle and intercept ([ref 7](https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Elassam.pdf)). Evaluate on a Raspberry Pi 5 to ensure ≥ 15 FPS.

- **Constructing a test set first** allows you to tune a classical algorithm via brute-force search over parameters or to train a lightweight model. Use optimisation techniques and measure both accuracy and runtime.

By systematically combining a diverse dataset, clear evaluation metrics and a parameter search, you can develop a horizon detection algorithm that meets the hackathon's accuracy and real-time constraints without resorting to heavy machine-learning models.

---

## References

1. [raw.githubusercontent.com – sallamander/horizon-detection/utils.py](https://raw.githubusercontent.com/sallamander/horizon-detection/master/utils.py)
2. [A fast horizon detector and a new annotated dataset for maritime video processing (arXiv)](https://arxiv.org/html/2110.13694v4)
3. [horizon_detection/myHorizon.py at master · k29/horizon_detection · GitHub](https://github.com/k29/horizon_detection/blob/master/myHorizon.py)
4. [paper_ettinger.pdf (University of Delaware)](https://www.eecis.udel.edu/~cer/arv/readings/paper_ettinger.pdf)
5. [Onboard Visual Horizon Detection for Unmanned Aerial Systems with Programmable Logic (MDPI)](https://www.mdpi.com/2079-9292/9/4/614)
6. [Ongoing project: horizon line detection – GCVG (Eötvös Loránd University)](https://cv.inf.elte.hu/index.php/2025/12/19/ongoing-project-horizon-line-detection/)
7. [Horizon line estimation (Elassam, LORIA seminar slides)](https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Elassam.pdf)
8. [raw.githubusercontent.com – NovelioPI/horizon-detection-for-uav README](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)
9. [Temporally Consistent Horizon Lines (University of Twente)](https://ris.utwente.nl/ws/files/254342269/yang_tem.pdf)
10. [raw.githubusercontent.com – NovelioPI/horizon-detection-for-uav README](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)
11. [Sky Segmentation Dataset – maadaa.ai](https://maadaa.ai/datasets/DatasetsDetail/Sky-Segmentation-Dataset)
12. [raw.githubusercontent.com – NovelioPI/horizon-detection-for-uav README](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)
13. [raw.githubusercontent.com – NovelioPI/horizon-detection-for-uav README](https://raw.githubusercontent.com/NovelioPI/horizon-detection-for-uav/master/README.md)

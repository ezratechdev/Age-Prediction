# Age Prediction

Age estimation from face images using OpenCV's DNN module with pretrained Caffe/TensorFlow models.

Faces are detected with an OpenCV SSD face detector, then each detected face is passed to the Levi & Hassner age classification network, which predicts one of 8 age brackets.

---

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) for dependency management
- `opencv-python < 5.0` (see [OpenCV version constraint](#opencv-version-constraint))

## Setup

### 1. Install dependencies

```bash
uv sync
```

This installs the exact versions pinned in `uv.lock`.

### 2. Download pretrained models

Model weights are **not tracked in this repository** (large binaries). Download them into `age_prediction_pretrained_models/` before running.

| File | Purpose | Source |
|------|---------|--------|
| `age_net.caffemodel` | Age classification weights (~44 MB) | [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel) |
| `age_deploy.prototxt` | Age network architecture | Tracked in repo |
| `opencv_face_detector_uint8.pb` | Face detector weights | [learnopencv/AgeGender](https://github.com/spmallick/learnopencv/tree/master/AgeGender) |
| `opencv_face_detector.pbtxt` | Face detector graph definition | Tracked in repo |

```bash
curl -L -o age_prediction_pretrained_models/age_net.caffemodel \
  https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
```

Verify the download is the real binary (~44 MB), not an HTML error page:

```bash
ls -lh age_prediction_pretrained_models/age_net.caffemodel
```

### 3. Add an input image

Place the image you want to analyse in `human_photo_to_predict/`. This directory is gitignored — input photos stay local and are never committed.

## Usage

```bash
uv run main.py
```

---

## Project structure

```
age_prediction/
├── age_prediction_pretrained_models/   # model weights + architecture files
│   ├── age_deploy.prototxt             # tracked
│   ├── opencv_face_detector.pbtxt      # tracked
│   ├── age_net.caffemodel              # gitignored — download separately
│   └── opencv_face_detector_uint8.pb   # gitignored — download separately
├── human_photo_to_predict/             # gitignored — local input images
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Age brackets

The age network is a **classifier, not a regressor**. It outputs a probability distribution over 8 fixed buckets:

```
(0-2)  (4-6)  (8-12)  (15-20)  (25-32)  (38-43)  (48-53)  (60-100)
```

There is no way to get a precise year estimate from this model — only the most probable bracket.

---

## Known limitations — accuracy needs improvement

**Test runs show the model is not accurate enough for production use.** Predictions are frequently off by one or more brackets, particularly for faces that fall near bracket boundaries. Treat all output as a rough estimate, not a reliable age measurement.

Contributing factors:

- **Age of the model.** The age network was trained on the Adience dataset (~26,500 images, published 2015). It does not reflect modern image quality, demographics, or camera characteristics.
- **Coarse, non-contiguous buckets.** The brackets skip ranges entirely — there is no class covering ages 3, 7, 13-14, 21-24, 33-37, etc. Faces in those gaps are forced into an adjacent bucket.
- **Known demographic bias.** Accuracy varies significantly across skin tone, gender presentation, and age group. The training set is not demographically balanced, and error rates are higher for underrepresented groups.
- **Sensitivity to input conditions.** Lighting, head pose, occlusion (glasses, facial hair, masks), image resolution, and makeup all degrade prediction quality.
- **Crop quality dependency.** The age network only sees the crop produced by the face detector. Poor or loose bounding boxes propagate error downstream.

### Improvement directions

- Add face alignment before the age crop (eye-landmark based) to normalise pose
- Add margin/padding around detected face boxes and tune the detector confidence threshold
- Aggregate predictions across multiple frames or augmented crops rather than trusting a single pass
- Report the full softmax distribution and a confidence value instead of only the top bracket
- Replace the backbone with a modern age estimation model (e.g. a regression-based model, or fine-tune on a current, more balanced dataset)
- Establish a labelled evaluation set and measure mean absolute error to quantify improvements instead of judging by eye

---

## OpenCV version constraint

This project requires **opencv-python < 5.0**.

OpenCV 5.0 removed `cv2.dnn.readNetFromCaffe()` and `cv2.dnn.readNetFromDarknet()`. On OpenCV 5.x, loading `age_net.caffemodel` fails with:

```
AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'
```

Two options:

1. **Stay on 4.x** (current approach) — pinned in `pyproject.toml` as `opencv-python<5.0`
2. **Migrate to ONNX** — convert the Caffe model to ONNX and load it with `cv2.dnn.readNetFromONNX()`, which unblocks upgrading to OpenCV 5

## Notes

- **No Google Colab dependency.** `google.colab.patches.cv2_imshow` is not installable outside Colab. Image display uses a local equivalent (matplotlib or `cv2.imshow`).
- **Model paths resolve relative to `main.py`**, not the current working directory, so the script runs correctly from any location.

## Responsible use

Age estimation models can be misused. This project is intended for [learning](https://www.geeksforgeeks.org/computer-vision/age-detection-using-deep-learning-in-opencv/) and experimentation. Do not use it for age verification, access control, surveillance, or any decision affecting a person's rights or access to services, the accuracy and bias limitations documented above make it unsuitable for those purposes.

Input photos are processed locally and are excluded from version control. If you use photos of other people, make sure you have their consent.
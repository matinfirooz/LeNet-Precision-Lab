# Requirement coverage and interpretation

Author: Matin Firoozbakht · https://github.com/matinfirooz

The original supplied PDF and README are preserved unchanged in this directory.

| Requirement | Implementation / evidence |
| --- | --- |
| Manual symmetric and asymmetric quantization | software/quantization.py; endpoint and integer-affine definitions kept separate |
| Provided sample | worked_example.json and corrected 197 code in the notebook |
| Manual INT16 / INT8 / INT4 accuracy | 18 NumPy-only full-test sweeps, two activation policies and three grids |
| Accuracy-versus-bit plot | manual_accuracy.png and pytorch_accuracy.png |
| Train LeNet-5 on Fashion-MNIST | 20 actual epochs; 54k/6k split; official 10k test |
| FP32 / FP16 / INT16 / INT8 / INT4 / INT2 | Saved checkpoints, full-test predictions and packed parameter images |
| Use PyTorch quantization tools | FakeQuantize/MinMaxObserver and a separate prepare/convert native INT8 path |
| Extract weights for hardware | All five learned layers exported with numerical and layout metadata |
| Trade-off and degradation questions | COMPLETE_ANSWERS.md sections 7–10; notebook sections 8–13 |
| Pruning title/introduction | Explicit additional magnitude-pruning and channel-compaction study |
| Submission identity | Matin Firoozbakht; GitHub @matinfirooz |

The source defines PTQ as conversion after FP32 training but later uses wording that could be read as training after conversion. This solution trains FP32, then evaluates PTQ without retraining. Recovery of pruned models is separately identified and never described as pure PTQ. No particular pruning rates or algorithms are specified; extension choices are fixed and reported.

The PDF does not prescribe an exact LeNet layer table, preprocessing policy or calibration size. The implemented dense C3/tanh/average-pooling variant, normalization, padding and splits are fully documented. Training/checkpoint selection and calibration do not use official test labels. The source's shifted min–max and integer-zero-point descriptions are numerically distinct; both are implemented and the sample arithmetic is corrected.

No student number or separate official report template was supplied. The named ZIP/notebook use HW4 and the author's full name. Transfer content to an official template and add a student number if the course separately requires them.

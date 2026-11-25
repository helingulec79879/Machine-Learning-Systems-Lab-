# Individual Project Proposal

**Student:** Helin Agze Gulec · **Student ID:** 79879 · **Course:** Machine Learning Systems · **Date:** 25.11.2026

**Title:** On-device waste-sorting classifier for a smart recycling bin

## 1. Problem
Mixed recyclables are often thrown into the wrong bin, which contaminates whole batches.
The goal is an image classifier that runs **on the bin itself** (Raspberry Pi 4 class CPU, no internet)
and tells the user, from one camera frame, which of 6 material classes the item belongs to.
On-device inference is required for privacy, offline use and low cost.

## 2. Dataset
**TrashNet** (Thung & Yang, Stanford): ~2,500 RGB photos, 6 classes – cardboard, glass, metal, paper, plastic, trash.
- Split: 70 / 15 / 15 % stratified train / val / test, fixed seed.
- Preprocessing: resize to 224×224 (also tested 160×160 for latency); augmentation – random crop, flip, colour jitter.
- Known issue: class imbalance ("trash" is the smallest class) → class-weighted loss, report macro-F1.

## 3. Success metrics
| Metric | Target |
|---|---|
| Top-1 accuracy (test) | ≥ 85 % |
| Macro-F1 (test) | ≥ 0.80 |
| Accuracy drop after optimization vs fp32 baseline | ≤ 2 percentage points |

## 4. Deployment constraints
| Constraint | Limit | How it is measured |
|---|---|---|
| Latency | **≤ 50 ms** per image, batch 1 | median of 200 runs after 20 warm-up runs, ONNX Runtime / TFLite on CPU (Raspberry Pi 4, or laptop CPU limited to 1 thread as a proxy) |
| Model size | **≤ 5 MB** | size of the exported model file on disk |

Size check: 5 MB ≈ 5.24 M parameters in int8, but only 1.31 M in fp32.

## 5. Model and planned optimization
- **Baseline:** MobileNetV3-Small pretrained on ImageNet, fine-tuned (≈ 1.5 M parameters with a 6-class head
  → ≈ 6 MB in fp32, so it **does not** meet the size limit yet).
- **Main optimization: post-training INT8 quantization** (static, calibrated on 200 training images) → ≈ 1.5 MB,
  expected 2–3× faster on CPU. If accuracy drops more than 2 pp → **quantization-aware training**.
- **Fallback / comparison:** knowledge distillation from a larger teacher (ResNet-50) into the small student.
- All experiments reuse the Lab 6 pipeline (`lab06/src`) (YAML config, seed, checkpoints, MLflow tracking).

## 6. Plan
| Week | Work |
|---|---|
| 1 | Data split, baseline fine-tuning, MLflow tracking |
| 2 | Measure fp32 latency and size; INT8 post-training quantization |
| 3 | QAT or distillation if needed; ablation (input resolution 224 vs 160) |
| 4 | Final evaluation on test set, latency report, write-up |

## 7. Risks
Small dataset → overfitting (mitigation: transfer learning, augmentation); latency on real hardware unknown
(mitigation: lower input resolution, 1-thread CPU proxy measurements early).

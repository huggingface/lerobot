# Torch.compile Benchmark Report: ACT

## Environment

- **Policy**: act
- **Device**: cuda
- **PyTorch**: 2.7.1+cu126
- **Dataset**: AdilZtn/grab_red_cube_test_25
- **Batch Size**: 8
- **Benchmark Parameters**: 100 inference runs, 50 training runs

## 🔧 Compilation Results

- **Status**: ✅ SUCCESS

## 🎯 Correctness Results

- **Status**: ❌ FAILED
- **Inference**: FAILED
- **Training**: FAILED

### Detailed Differences

- **Max Action Difference**: 5.57e-02 (threshold: 1.00e-05)
- **Loss Difference**: 8.93e-05 (threshold: 1.00e-05)

### ⚠️ Correctness Analysis

- **Action diff magnitude**: 5.57e-02 (SEVERE)
- **Loss diff magnitude**: 8.93e-05 (MINOR)
- **Likely causes**: Graph breaks, dynamic shapes, numerical precision issues

## ⚡ Performance Results

### Inference Performance

- **Original**: 21.75 ms/iter
- **Compiled**: 21.46 ms/iter
- **🚀 Speedup**: 1.01x (⚠️ INSUFFICIENT)

### Training Performance

- **Original**: 68.59 ms/iter
- **Compiled**: 61.15 ms/iter
- **🚀 Speedup**: 1.12x

### Consistency Metrics

- **Average Loss Difference**: 4.87e-03
- **Average Grad Norm Difference**: 1.60e+00

## 📋 Success Criteria Analysis

- **✅ Compilation**: PASSED
- **✅ Correctness**: FAILED
- **✅ Performance**: FAILED
- **✅ Benchmarking**: PASSED

## 🎯 Overall Result

❌ NEEDS WORK: torch.compile not yet functional

## 🛠️ Next Steps

1. **Debug numerical differences** - Check for precision issues
2. **Verify tensor operations** - Ensure deterministic behavior
3. **Test with smaller tolerance** - May be acceptable for some use cases

## 🔍 Raw Data

```json
{
  "success": false,
  "policy": "act",
  "device": "cuda",
  "pytorch_version": "2.7.1+cu126",
  "compilation_successful": true,
  "compilation_error": null,
  "correctness": {
    "inference_correct": false,
    "training_correct": false,
    "action_diff": 0.05568695068359375,
    "loss_diff": 8.934736251831055e-5
  },
  "correctness_passed": false,
  "inference_benchmarked": true,
  "training_benchmarked": true,
  "time_original_inference": 21.745667571667582,
  "time_compiled_inference": 21.46407115040347,
  "speedup_inference": 1.0131194319703334,
  "time_original_training": 68.5850445041433,
  "time_compiled_training": 61.15469123702496,
  "speedup_training": 1.1215009530228772,
  "loss_consistency": 0.004866420030593872,
  "grad_norm_consistency": 1.599840692281723
}
```

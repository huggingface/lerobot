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
- **Inference**: PASSED
- **Training**: FAILED

### Detailed Differences

- **Max Action Difference**: 0.00e+00 (threshold: 1.00e-05)
- **Loss Difference**: 8.93e-05 (threshold: 1.00e-05)

### ⚠️ Correctness Analysis

- **Action diff magnitude**: 0.00e+00 (MINOR)
- **Loss diff magnitude**: 8.93e-05 (MINOR)
- **Likely causes**: Graph breaks, dynamic shapes, numerical precision issues

## ⚡ Performance Results

### Inference Performance

- **Original**: 22.32 ms/iter
- **Compiled**: 21.48 ms/iter
- **🚀 Speedup**: 1.04x (⚠️ INSUFFICIENT)

### Training Performance

- **Original**: 72.74 ms/iter
- **Compiled**: 63.96 ms/iter
- **🚀 Speedup**: 1.14x

### Consistency Metrics

- **Average Loss Difference**: 7.22e-03
- **Average Grad Norm Difference**: 1.68e+00

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
    "inference_correct": true,
    "training_correct": false,
    "action_diff": 0.0,
    "loss_diff": 8.934736251831055e-5
  },
  "correctness_passed": false,
  "inference_benchmarked": true,
  "training_benchmarked": true,
  "time_original_inference": 22.322929948568344,
  "time_compiled_inference": 21.476867329329252,
  "speedup_inference": 1.0393941353860157,
  "time_original_training": 72.73679859936237,
  "time_compiled_training": 63.96130052395164,
  "speedup_training": 1.137200119502332,
  "loss_consistency": 0.007216747999191284,
  "grad_norm_consistency": 1.6769276988506316
}
```

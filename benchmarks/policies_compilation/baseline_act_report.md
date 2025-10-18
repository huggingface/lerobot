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

- **Original**: 22.14 ms/iter
- **Compiled**: 22.95 ms/iter
- **🚀 Speedup**: 0.96x (⚠️ SLOWDOWN)

### Training Performance

- **Original**: 68.74 ms/iter
- **Compiled**: 61.31 ms/iter
- **🚀 Speedup**: 1.12x

### Consistency Metrics

- **Average Loss Difference**: 6.78e-03
- **Average Grad Norm Difference**: 1.53e+00

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
  "time_original_inference": 22.140723338816315,
  "time_compiled_inference": 22.948127458803356,
  "speedup_inference": 0.9648161218628187,
  "time_original_training": 68.73527298215777,
  "time_compiled_training": 61.31022967863828,
  "speedup_training": 1.1211061081068912,
  "loss_consistency": 0.0067824447154998775,
  "grad_norm_consistency": 1.5347092628479004
}
```

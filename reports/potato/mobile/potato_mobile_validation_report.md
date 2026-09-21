# Potato Mobile Prototype Package Validation Report

**Validated Artifact:** `supervised_mobilenetv3_float16.tflite`
**Size:** 5.76 MB
**SHA-256 Checksum:** `f3b3620ea3363f5503da92d89a41ed7316b242d4447015bf59ec5efd416ad83c`

## 1. Technical Compliance Checklist

- [x] Model checksum matches package manifest
- [x] Class labels order strictly verified: `early_blight`, `healthy`, `late_blight`
- [x] Preprocessing contract: `(224, 224, 3)` uint8 neutral fill (114, 114, 114)
- [x] 100% categorical agreement with source Keras model verified
- [x] 3-stage safe abstention engine operational

## 2. Abstention and Margin Performance

- Tested on 50 test samples: 50 confident, 0 abstained.
- Selective accuracy under confidence gating: 100.00%

현재 SafeDiff-VLA / SmolVLA 기반 코드의 아키텍처를 **수정하지 말고 audit만 수행**해 주세요.

이번 작업의 목적은 코드를 고치는 것이 아니라, 현재 data flow와 model architecture를 정확히 분석해서 아래 질문에 답하는 것입니다.

핵심 질문:

> 현재 구조가 왜 `SmolVLA nominal action → refinement planner → final action` 형태가 되었는지, 그리고 이를
> `VLM/VLA latent + current state (+ optional predicted future state) → temporal action decoder → action trajectory`
> 형태로 바꾸는 것이 구조적으로 타당한지 검증해 주세요.

중요: **이번 단계에서는 어떤 코드도 수정하지 마세요.**
리팩터링, patch, 새 class 구현, config 변경도 하지 마세요.
오직 분석과 보고만 수행하세요.

---

## 1. 현재 전체 data flow를 실제 코드 기준으로 추적

반드시 코드에서 실제 call path를 따라가며 다음 흐름을 추적해 주세요.

```text
observation
(image / language / robot state)
        ↓
preprocessing
        ↓
SmolVLA backbone / processor
        ↓
multimodal representation
        ↓
action generation
        ↓
nominal action chunk
        ↓
SafeDiff / refinement / gating
        ↓
executed action
```

각 단계에 대해 다음을 명시하세요.

* 파일명
* class명
* function명
* 입력 tensor
* 출력 tensor
* tensor shape
* trainable/frozen 여부
* 다음 module로 무엇이 전달되는지

최종 보고서에는 실제 코드 기준으로 **end-to-end call graph**를 만들어 주세요.

---

## 2. SmolVLA가 action을 정확히 어떻게 만드는지 분석

다음 질문에 답해 주세요.

### A. Vision 처리

* camera image가 어떤 vision encoder를 통과하는가?
* 여러 camera가 있다면 어떻게 결합되는가?
* image feature의 shape은 무엇인가?

### B. Language 처리

* instruction/token이 어디에서 embedding 되는가?
* visual feature와 언제 fusion되는가?

### C. Robot state

* current robot state가 SmolVLA 내부에서 사용되는가?
* 사용된다면 어느 시점에 어떤 방식으로 들어가는가?
* projection / tokenization / concatenation / conditioning 방식은 무엇인가?
* state가 전혀 안 들어가는 path가 있는가?

### D. Multimodal latent

가장 중요합니다.

SmolVLA에서 **최종 action output 이전에 존재하는 multimodal hidden representation**을 찾으세요.

예:

```text
[B, N_tokens, D]
```

또는 이에 해당하는 representation.

다음을 확인하세요.

* 이 latent가 vision 정보를 포함하는가?
* language 정보를 포함하는가?
* current robot state도 포함하는가?
* 어떤 token이 어떤 modality에 대응하는가?
* action generation 직전 representation인가?

가능하면 정확한 tensor shape과 생성 위치를 제시하세요.

---

## 3. 기존 action head 분석

SmolVLA의 기존 action head가 정확히 무엇을 하는지 분석하세요.

다음 중 어떤 구조인지 확인하세요.

* autoregressive decoder
* flow matching
* diffusion-like head
* deterministic regression
* transformer action expert
* 기타

그리고 반드시 다음에 답하세요.

* 50-step action trajectory가 한 번에 생성되는가?
* timestep별 독립 예측인가?
* horizon 축에 temporal mixing이 있는가?
* action query/token이 존재하는가?
* current state가 action decoding 단계에 직접 영향을 주는가?
* action head가 multimodal latent와 어떻게 interaction하는가?

최종적으로 수식 또는 pseudo-code 수준으로 설명해 주세요.

예:

```python
latent = ...
action_queries = ...
actions = action_head(latent, state, ...)
```

---

## 4. `nominal action`의 정확한 의미 확인

현재 코드에서 nominal action이 정확히 무엇인지 정의하세요.

다음을 확인하세요.

* SmolVLA original action head의 raw output인가?
* normalization 전/후인가?
* environment action space로 변환된 뒤인가?
* chunk shape은 정확히 무엇인가?
* `[B, 50, action_dim]`인가?
* inference 중 50 step 전체가 그대로 실행되는가?

그리고 nominal이 생성된 뒤 어떤 module들이 추가로 이를 수정하는지 추적하세요.

---

## 5. 현재 SafeDiff refinement 구조 분석

다음을 각각 분석하세요.

### ConditionalDiffusionPlanner

* 정확한 입력은 무엇인가?
* nominal action이 필수 입력인가?
* VLM latent를 직접 받는가?
* current state를 직접 받는가?
* predicted future state를 받는가?
* diffusion timestep embedding은 어떻게 들어가는가?
* horizon 축 temporal mixing이 존재하는가?

특히 다음을 코드 레벨로 확인하세요.

```python
nn.Linear([B, H, D])
```

형태의 연산이 horizon을 mixing하는 것으로 잘못 해석되어 있지는 않은지 확인하세요.

Conv1D / attention / RNN / temporal Transformer 등이 없다면 명시적으로:

> 각 horizon timestep이 독립 처리된다.

라고 결론 내려 주세요.

### Temporal residual planner

새로 구현된 `temporal_residual` 구조가 있다면 이것도 동일하게 분석하세요.

* input
* output
* residual definition
* temporal mixing
* state conditioning
* nominal dependency
* final action formula

예:

```text
a_final = a_nominal + scale * delta_a
```

인지 확인하세요.

---

## 6. Gating / replanning 분석

현재 gate가 정확히 무엇을 하는지 분석하세요.

* 언제 발동하는가?
* action chunk를 중간에 끊는가?
* 새 nominal trajectory를 다시 생성하는가?
* refinement를 다시 실행하는가?
* execute horizon과 action horizon의 관계는 무엇인가?

특히 다음 설정들을 코드 기준으로 비교하세요.

```text
action_horizon = 16
execute_horizon = 4
```

vs

```text
action_horizon = 50
execute_horizon = 50
```

SmolVLA original checkpoint가 기대하는 action chunk length와 현재 SafeDiff config가 일치하는지 확인하세요.

---

## 7. 우리가 제안한 새 구조가 실제로 타당한지 검증

아래 구조를 코드 관점에서 비판적으로 평가해 주세요.

```text
Image + Language
       ↓
SmolVLA / VLM Backbone
       ↓
Multimodal Latent
       │
Current State
       │
(+ optional Future State)
       ↓
Temporal Action Decoder
       ↓
50-step Action Trajectory
```

즉:

```text
VLM latent
+ current state
(+ predicted future state)
→ temporal action decoder
→ action trajectory
```

입니다.

이 구조가 실제 SmolVLA architecture와 잘 맞는지 평가하세요.

반드시 아래 질문 각각에 답하세요.

### Q1

SmolVLA의 action head 직전 latent를 별도의 temporal action decoder에 conditioning하는 것이 가능한가?

### Q2

그 latent에 이미 current state가 포함되어 있다면, current state를 decoder에 다시 명시적으로 넣는 것이:

* 필요함
* 유용할 수 있음
* 중복임
  중 어느 것인지 판단하고 이유를 설명하세요.

### Q3

기존 SmolVLA action head를 완전히 대체하는 것이 합리적인가?

아니면 기존 action head 일부를 재사용하는 것이 더 합리적인가?

### Q4

SmolVLA backbone을 frozen encoder처럼 사용하고 새 decoder만 학습하는 것이 기술적으로 가능한가?

gradient flow 관점에서 확인하세요.

### Q5

새 decoder가 action trajectory 전체를 직접 생성하는 것이 기존 nominal-refinement 구조보다 더 자연스러운가?

반론이 있다면 반드시 제시하세요.

---

## 8. Decoder architecture 후보 비교

새 구조를 쓴다고 가정할 때 다음 후보를 비교하세요.

### A. Transformer decoder

```text
learned action queries [H,D]
→ self-attention across horizon
→ cross-attention to VLM latent
→ state conditioning
→ action projection
```

### B. Temporal 1D Conv / Temporal U-Net

```text
action queries / latent features
→ temporal conv blocks
→ conditioned on VLM/state
→ actions
```

### C. 기존 SmolVLA action expert 일부 재사용

기존 action-generation architecture가 이미 잘 설계되어 있다면:

* 어떤 부분을 재사용할 수 있는지
* 무엇만 교체하면 되는지

비교하세요.

각 후보에 대해:

* 구현 복잡도
* temporal coherence
* 기존 checkpoint 활용성
* 학습 안정성
* VLABench 적용 적합성
* 예상 GPU 비용

을 비교해 주세요.

---

## 9. Diffusion이 실제로 필요한지도 검증

다음 두 설계를 별도로 평가하세요.

```text
VLM latent + state
→ deterministic temporal decoder
→ actions
```

vs

```text
VLM latent + state
→ temporal diffusion decoder
→ actions
```

다음에 답하세요.

* 이 task에서 diffusion이 반드시 필요한 근거가 있는가?
* multi-modality가 실제로 중요한가?
* deterministic decoder를 먼저 검증하는 것이 합리적인가?
* diffusion을 쓴다면 nominal refinement가 아니라 direct conditional generation으로 쓰는 게 더 맞는가?

---

## 10. Predicted future state의 역할 평가

현재 state predictor가 있다면 아래를 확인하세요.

* 입력
* 출력
* supervision
* prediction horizon
* 실제 prediction quality를 평가하는 metric 존재 여부

그리고 future state를 decoder에 넣는 것이 실제로 도움이 될지 평가하세요.

특히 다음 위험성을 검토하세요.

```text
prediction error
→ action decoder conditioning error
→ compounding failure
```

따라서 다음 두 구조 중 어떤 것을 먼저 검증해야 하는지 판단하세요.

```text
A. VLM latent + current state → action decoder
B. VLM latent + current state + future state → action decoder
```

---

## 11. Supervision / action representation audit

새 decoder가 학습해야 할 target이 정확히 무엇인지 확인하세요.

* absolute action?
* delta action?
* joint position?
* end-effector delta?
* gripper?
* normalized action?

dataset의 target representation과 simulator가 받는 action representation 사이의 변환을 정확히 추적하세요.

다음을 반드시 확인하세요.

```text
dataset action
→ normalization
→ model target
→ model prediction
→ unnormalization
→ simulator action
```

여기에서 shape/order/scale mismatch 가능성이 있는지도 확인하세요.

---

## 12. 현재 실험 결과와 architecture가 논리적으로 일치하는지 평가

현재 관측된 결과는 다음과 같습니다.

```text
baseline SmolVLA               ≈ 15%
pure nominal, gate ON          ≈ 10%
pure nominal, gate OFF         ≈ 25%
diffusion refinement           = 0% consistently
temporal residual refinement   ≈ 10%
```

이 결과가 코드 구조와 논리적으로 일치하는지 분석하세요.

특히:

* 왜 pure nominal이 가장 잘 나올 수 있는가?
* 왜 refinement가 성능을 깎을 수 있는가?
* diffusion 0%가 temporal independence와 일치하는가?
* temporal residual이 10%로 회복된 것이 temporal mixing 때문이라는 해석이 가능한가?
* 그러나 nominal 25%보다 낮다는 사실은 무엇을 의미하는가?

과도하게 확신하지 말고 대안 가설도 함께 제시하세요.

---

## 13. Architecture red flags

현재 코드에서 architecture 관점의 red flag를 모두 찾아 목록화하세요.

예:

* action head output을 다시 planner input으로 사용하는 구조
* horizon timestep 독립 처리
* state가 action decoder에 직접 안 들어감
* normalization mismatch 가능성
* action horizon mismatch
* frozen backbone에서 원하는 latent가 제대로 노출되지 않음
* future-state predictor가 noisy conditioning으로 사용됨
* gate가 strong nominal trajectory를 지나치게 자주 끊음

각 항목마다 severity를:

```text
critical
high
medium
low
```

로 분류하세요.

---

## 14. 최종 결론

마지막에는 반드시 아래 형식으로 결론을 내려 주세요.

### A. 현재 architecture 요약

현재 코드가 실제로 무엇을 하는지 5~10줄로 정리.

### B. 가장 큰 architecture 문제 3개

우선순위 순서대로.

### C. 새 구조에 대한 판정

아래 중 하나를 선택하세요.

```text
1. Strongly recommended
2. Reasonable, but needs modifications
3. Not recommended
```

대상 구조:

```text
VLM latent + current state
(+ optional predicted future state)
→ temporal action decoder
→ action trajectory
```

### D. 권장 final architecture

가장 적합한 구조를 ASCII diagram으로 제시하세요.

### E. 첫 implementation에서 반드시 포함할 것

최대 5개.

### F. 첫 implementation에서 절대 넣지 말아야 할 것

최대 5개.

### G. 최소 검증 실험

full training 전에 반드시 해야 하는 최소 실험을 제시하세요.

예:

```text
1. tensor shape check
2. gradient check
3. 100-sample overfit
4. action distribution check
5. short 2k~5k training
6. small fixed-seed eval
```

---

## 매우 중요한 제약

이번 audit에서는:

* 코드를 수정하지 마세요.
* 새 파일을 만들지 마세요.
* config를 변경하지 마세요.
* training을 돌리지 마세요.
* 긴 eval을 돌리지 마세요.
* 추측으로 architecture를 설명하지 마세요.

반드시 **실제 repository code를 읽고 call path를 추적한 결과만** 근거로 결론을 내리세요.

확실하지 않은 부분은:

```text
CONFIRMED
LIKELY
UNKNOWN
```

중 하나로 표시하세요.

최종 보고서는 다른 연구자 또는 다른 AI가 그대로 읽고 architecture correctness를 재검증할 수 있을 정도로 구체적으로 작성해 주세요.

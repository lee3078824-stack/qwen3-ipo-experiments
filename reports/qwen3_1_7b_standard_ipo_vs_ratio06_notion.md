# Qwen3-1.7B Standard IPO vs Fixed Ratio IPO 실험 정리

## 1. 실험 개요

이번 실험에서는 `Qwen/Qwen3-1.7B` 모델을 대상으로 기존 `Standard IPO`와 `target_ratio=0.6`으로 고정한 `Fixed target-ratio IPO`를 비교했다.

핵심 질문은 다음과 같다.

- 기존 IPO처럼 고정된 IPO target gap을 사용하는 것이 좋은가?
- 아니면 chosen을 어느 정도 선호할지 직접 `target_ratio`로 지정하는 방식이 더 안정적인가?
- `target_ratio=0.6`처럼 완만한 preference 강도를 주었을 때 benchmark 성능이 유지되는가?

## 2. 방법 비교

| Method | 설명 |
| --- | --- |
| Standard IPO | 기존 IPO objective를 그대로 사용 |
| Fixed target-ratio IPO | IPO의 target gap을 `target_ratio=0.6`에 맞게 직접 설정 |

Standard IPO는 IPO가 정의한 고정 target gap에 policy와 reference의 preference gap 차이를 맞추는 방식이다.

Fixed target-ratio IPO는 target gap을 직접 설정한다. 이번 실험에서는 chosen response를 `60%`, rejected response를 `40%` 정도로 선호하도록 목표를 설정했다.

```text
target_ratio = P(chosen) = 0.6
P(rejected) = 1 - target_ratio = 0.4
target_gap = logit(target_ratio)

loss = (beta * gap - target_gap)^2
```

여기서 `target_ratio=0.6`은 chosen을 rejected보다 선호하되, 너무 강하게 한쪽으로 밀지 않는 완만한 preference 강도를 의미한다.

## 3. 실험 세팅

| 항목 | Standard IPO | Fixed target-ratio IPO |
| --- | --- | --- |
| Model | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B |
| Dataset | Jennny/helpsteer2-helpfulness-preference | Jennny/helpsteer2-helpfulness-preference |
| Fine-tuning | Full fine-tuning | Full fine-tuning |
| Precision | bfloat16 | bfloat16 |
| Objective | standard_ipo | target_ratio |
| Target ratio | 사용하지 않음 | 0.6 |
| Length normalization | none | none |
| beta | 0.01 | 0.01 |
| learning rate | 5e-7 | 5e-7 |
| epoch | 3 | 3 |
| total steps | 2709 | 2709 |
| batch | per_device_train_batch_size=1 | per_device_train_batch_size=1 |
| grad accumulation | 8 | 8 |

## 4. 학습 결과

Standard IPO와 fixed target-ratio IPO는 loss 정의가 다르기 때문에 loss 값을 직접 비교하는 것은 적절하지 않다.

따라서 학습 결과는 다음 지표를 중심으로 보는 것이 좋다.

- actual ratio
- reward accuracy
- reward margin

| 항목 | Standard IPO | ratio=0.6 IPO |
| --- | ---: | ---: |
| Final train loss | 631.0268 | 0.0324 |
| Final eval loss | 2065.6167 | 0.1352 |
| Train actual ratio | 0.5778 | 0.5695 |
| Eval actual ratio | 0.5222 | 0.5185 |
| Train reward accuracy | 0.9500 | 0.9750 |
| Eval reward accuracy | 0.6783 | 0.6863 |
| Train reward margin | 0.3155 | 0.2809 |
| Eval reward margin | 0.0899 | 0.0749 |

## 5. 학습 결과 해석

- `ratio=0.6 IPO`는 train reward accuracy가 `0.9750`으로 Standard IPO의 `0.9500`보다 높았다.
- eval reward accuracy도 `0.6863`으로 Standard IPO의 `0.6783`보다 소폭 높았다.
- 즉 chosen/rejected pair를 맞히는 정확도 기준으로는 `ratio=0.6 IPO`가 약간 더 좋았다.
- 반면 reward margin은 Standard IPO가 더 컸다.
    - Standard IPO eval margin: `0.0899`
    - ratio=0.6 IPO eval margin: `0.0749`
- 이는 Standard IPO가 chosen과 rejected 사이의 점수 차이를 더 크게 벌렸다는 의미다.
- 반대로 `ratio=0.6 IPO`는 너무 큰 margin을 만들기보다는 완만한 preference 강도를 유지한 것으로 볼 수 있다.
- eval actual ratio는 두 모델 모두 약 `0.52` 근처에 머물렀다.
- 즉 validation pair에서는 target ratio가 강하게 일반화되지는 않았지만, reward accuracy는 ratio=0.6 쪽이 조금 더 높았다.

## 6. Benchmark 결과

| Task | Metric | Standard IPO | ratio=0.6 IPO | 차이 |
| --- | --- | ---: | ---: | ---: |
| ARC-Challenge | acc_norm | 0.4309 | 0.4300 | -0.0009 |
| ARC-Easy | acc_norm | 0.6982 | 0.7020 | +0.0038 |
| HellaSwag | acc_norm | 0.6091 | 0.6081 | -0.0010 |
| TruthfulQA MC2 | acc | 0.4610 | 0.4614 | +0.0005 |
| Winogrande | acc | 0.6069 | 0.6125 | +0.0055 |
| Mean | - | 0.5612 | 0.5628 | +0.0016 |

## 7. Task별 관찰

### ARC-Challenge

- Standard IPO: `0.4309`
- ratio=0.6 IPO: `0.4300`
- 차이: `-0.0009`
- 두 모델의 차이는 거의 없으며, Standard IPO가 아주 소폭 높았다.

### ARC-Easy

- Standard IPO: `0.6982`
- ratio=0.6 IPO: `0.7020`
- 차이: `+0.0038`
- ratio=0.6 IPO가 소폭 우세했다.

### HellaSwag

- Standard IPO: `0.6091`
- ratio=0.6 IPO: `0.6081`
- 차이: `-0.0010`
- Standard IPO가 아주 소폭 높았지만, 사실상 거의 동일한 수준이다.

### TruthfulQA MC2

- Standard IPO: `0.4610`
- ratio=0.6 IPO: `0.4614`
- 차이: `+0.0005`
- 차이는 매우 작지만 ratio=0.6 IPO가 약간 높았다.

### Winogrande

- Standard IPO: `0.6069`
- ratio=0.6 IPO: `0.6125`
- 차이: `+0.0055`
- ratio=0.6 IPO가 가장 뚜렷하게 우세한 task였다.

## 8. 종합 해석

- Qwen3-1.7B에서는 `ratio=0.6 IPO`가 Standard IPO보다 평균 benchmark 점수에서 소폭 높았다.
- 평균 차이는 `+0.0016`으로 매우 작다.
- 따라서 큰 성능 향상이라고 보기는 어렵고, Standard IPO와 거의 동등한 성능을 유지한 것으로 보는 것이 적절하다.
- 다만 `ratio=0.6 IPO`는 ARC-Easy, TruthfulQA MC2, Winogrande에서 Standard IPO보다 높았다.
- Standard IPO는 ARC-Challenge와 HellaSwag에서 소폭 높았다.
- 학습 지표에서는 ratio=0.6 IPO가 reward accuracy는 더 높고, Standard IPO는 reward margin이 더 높았다.
- 이는 ratio=0.6 IPO가 더 완만한 preference 강도를 유지하면서도 benchmark 성능을 유지했다는 점에서 의미가 있다.

## 9. 결론

- Qwen3-1.7B 실험에서 `target_ratio=0.6 IPO`는 Standard IPO와 거의 동등한 benchmark 성능을 보였다.
- 평균 benchmark 기준으로는 `ratio=0.6 IPO`가 Standard IPO보다 `+0.0016` 높았다.
- 그러나 차이가 매우 작기 때문에 성능 향상보다는 preference 강도 제어 가능성에 더 큰 의미가 있다.
- Standard IPO는 reward margin을 더 크게 만들었고, ratio=0.6 IPO는 reward accuracy가 더 높았다.
- 따라서 `ratio=0.6 IPO`는 chosen을 과도하게 밀지 않으면서도 Standard IPO 수준의 성능을 유지하는 안정적인 설정으로 볼 수 있다.

최종적으로 Qwen3-1.7B에서는 `target_ratio=0.6` 고정 방식이 Standard IPO의 성능을 크게 해치지 않으면서 preference 강도를 명시적으로 제어할 수 있는 대안으로 작동했다.

# prototype.py
# ============================================================
# VLA(Vision-Language-Action) Failure Detection 미니 프로토타입
#
# implementation_plan.md 의 설계를 numpy 만으로 구현한 self-contained 버전입니다.
# - RACER  : 정책 네트워크 + 언어 모델 + 실패 감지/인간 개입 복구 루프
# - TriDet : SGP 레이어 + Trident-head (시간적 행동 탐지 모사)
# - Multimodal Object Detection : 시각 + 언어 특징 결합
#
# 외부 데이터/딥러닝 프레임워크 없이 mock 데이터로 전체 흐름을 시연합니다.
# 실행: python prototype.py
# ============================================================

from __future__ import annotations

import numpy as np

# 재현 가능한 결과를 위한 시드 고정
rng = np.random.default_rng(42)


# ------------------------------------------------------------
# 1. RACER: 정책 네트워크 (policy_network.py 대체)
# ------------------------------------------------------------
class PolicyNetwork:
    """2층 MLP 정책 네트워크. 상태 -> 행동 로짓."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # He 초기화와 유사한 스케일의 무작위 가중치 (mock)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2  # 행동 로짓

    def action_probs(self, x: np.ndarray) -> np.ndarray:
        logits = self(x)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)  # softmax


# ------------------------------------------------------------
# 2. RACER: 언어 모델 (language_model.py 대체)
# ------------------------------------------------------------
class LanguageModel:
    """텍스트를 고정 차원 임베딩으로 바꾸는 mock 언어 모델 (해시 기반)."""

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim

    def forward(self, input_text: str) -> np.ndarray:
        # 단어별 해시 시드로 벡터를 만들어 평균 (실제 LM의 임베딩을 모사)
        if not input_text.strip():
            raise ValueError("빈 텍스트는 임베딩할 수 없습니다.")
        vectors = []
        for token in input_text.split():
            token_rng = np.random.default_rng(abs(hash(token)) % (2**32))
            vectors.append(token_rng.normal(0, 1, self.embed_dim))
        return np.mean(vectors, axis=0, keepdims=True)


# ------------------------------------------------------------
# 3. RACER: 실패 감지 및 인간/언어 개입 (human_interception.py 대체)
# ------------------------------------------------------------
def detect_failure(action_probs: np.ndarray, threshold: float = 0.7) -> bool:
    """정책 확신도가 낮으면(최대 확률 < threshold) 실패 위험으로 판단."""
    return float(action_probs.max()) < threshold


def get_human_input() -> str:
    """인간(또는 상위 VLM)의 복구 지시를 모사."""
    return "그리퍼를 5cm 위로 올린 뒤 물체를 다시 잡으세요."


def update_state_with_human_input(state: np.ndarray, human_text: str, lm: LanguageModel) -> np.ndarray:
    """복구 지시 임베딩을 상태에 반영해 새로운 상태를 만든다."""
    guidance = lm.forward(human_text)
    # 임베딩을 상태 차원에 맞게 반복/절단하여 더해줌 (mock 결합)
    tiled = np.resize(guidance, state.shape)
    return state + 0.5 * tiled


# ------------------------------------------------------------
# 4. TriDet: SGP 레이어 (sgp_layer.py 대체)
# ------------------------------------------------------------
class SGPLayer:
    """비디오 특징 시퀀스에서 지역(local)+전역(global) 정보를 결합하는 mock SGP."""

    def __init__(self, input_size: int, time_range: int):
        self.input_size = input_size
        self.time_range = time_range

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        # sequence: (T, D) 비디오 프레임 특징
        if sequence.ndim != 2:
            raise ValueError(f"(T, D) 형태의 시퀀스가 필요합니다. 입력: {sequence.shape}")
        local_feat = np.convolve(sequence.mean(axis=1), np.ones(3) / 3, mode="same")  # 지역 평균
        global_feat = sequence.mean()  # 전역 풀링
        return sequence * local_feat[:, None] + global_feat  # (T, D)

    __call__ = forward


# ------------------------------------------------------------
# 5. TriDet: Trident-head (trident_head.py 대체)
# ------------------------------------------------------------
class TridentHead:
    """행동 구간의 시작/중심/끝 경계를 추정하는 mock Trident-head."""

    def __init__(self, input_size: int):
        self.w_start = rng.normal(0, 0.1, input_size)
        self.w_end = rng.normal(0, 0.1, input_size)
        self.w_center = rng.normal(0, 0.1, input_size)

    def forward(self, features: np.ndarray) -> dict:
        # features: (T, D) -> 프레임별 시작/끝/중심 점수
        start_scores = features @ self.w_start
        end_scores = features @ self.w_end
        center_scores = features @ self.w_center
        return {
            "start": int(start_scores.argmax()),
            "end": int(end_scores.argmax()),
            "center": int(center_scores.argmax()),
        }

    __call__ = forward


# ------------------------------------------------------------
# 6. Multimodal Object Detection (multimodal_model.py 대체)
# ------------------------------------------------------------
class MultimodalModel:
    """시각 특징과 언어 임베딩을 결합하는 mock 멀티모달 모델."""

    def __init__(self, visual_dim: int, language_model: LanguageModel):
        self.visual_dim = visual_dim
        self.lm = language_model

    def forward(self, input_image: np.ndarray, input_text: str) -> np.ndarray:
        if input_image.shape[-1] != self.visual_dim:
            raise ValueError(f"이미지 특징 차원 불일치: {input_image.shape[-1]} != {self.visual_dim}")
        text_embedding = self.lm.forward(input_text)
        return np.concatenate([input_image, text_embedding], axis=1)  # (1, visual+text)


# ------------------------------------------------------------
# 메인 데모: 실패 감지 -> 언어 지시 개입 -> 복구
# ------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("VLA Failure Detection 프로토타입 데모 (mock data)")
    print("=" * 60)

    policy_net = PolicyNetwork(input_size=100, hidden_size=50, output_size=2)
    language_model = LanguageModel(embed_dim=64)
    sgp_layer = SGPLayer(input_size=100, time_range=30)
    trident_head = TridentHead(input_size=100)
    multimodal_model = MultimodalModel(visual_dim=256, language_model=language_model)

    try:
        # --- [1] RACER: 정책 실행 및 실패 감지 ---
        state = rng.random((1, 100))  # 로봇 상태 (mock)
        probs = policy_net.action_probs(state)
        print(f"\n[1] 정책 행동 확률: {np.round(probs, 3)}")

        if detect_failure(probs, threshold=0.7):
            print("    -> 확신도 낮음: 실패 위험 감지!")
            human_text = get_human_input()
            print(f"    -> 복구 지시 수신: '{human_text}'")
            state = update_state_with_human_input(state, human_text, language_model)
            probs = policy_net.action_probs(state)
            print(f"    -> 복구 후 행동 확률: {np.round(probs, 3)}")
        else:
            print("    -> 정상: 실패 위험 없음")

        # --- [1b] 강제 실패 시나리오: 높은 임계값으로 개입 흐름 시연 ---
        print("\n[1b] 강제 실패 시나리오 (threshold=0.95)")
        if detect_failure(probs, threshold=0.95):
            print("    -> 확신도 낮음: 실패 위험 감지!")
            human_text = get_human_input()
            print(f"    -> 복구 지시 수신: '{human_text}'")
            state = update_state_with_human_input(state, human_text, language_model)
            probs = policy_net.action_probs(state)
            print(f"    -> 복구 후 행동 확률: {np.round(probs, 3)}")

        # --- [2] TriDet: 비디오 시퀀스에서 행동(실패 순간) 구간 탐지 ---
        video_features = rng.random((30, 100))  # 30프레임 (mock)
        refined = sgp_layer(video_features)
        boundaries = trident_head(refined)
        print(f"\n[2] TriDet 행동 구간 추정: 시작={boundaries['start']}프레임, "
              f"중심={boundaries['center']}프레임, 끝={boundaries['end']}프레임")

        # --- [3] 멀티모달 결합: 이미지 + 텍스트 지시 ---
        image_features = rng.random((1, 256))  # 이미지 특징 (mock)
        instruction = "로봇이 물건을 집어넣으세요."
        combined = multimodal_model.forward(image_features, instruction)
        print(f"\n[3] 멀티모달 결합 특징 shape: {combined.shape} "
              f"(시각 256 + 언어 64)")

        print("\n데모 완료: 실패 감지 -> 언어 개입 -> 복구 흐름이 정상 동작했습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

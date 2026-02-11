# 크로스 엔트로피가 크로스 엔트로피라는 이름을 갖게 된 이유

손실함수는 실제값과 예측값 사이의 오차에 대한 식으로 "모델의 예측이 정답이랑 얼마나 어긋났는가?"를 나타낸다.

엔트로피는 "확률 분포가 얼마나 불확실한가"를 수치로 나타낸 개념으로 하나의 분포에 대해 불확실성 또는 정보량을 나타낸다.

- 어떤 사건이 예측하기 어려울수록 엔트로피가 크다 (예측이 쉬우면 엔트로피가 작다)

라는 특징이 있다

크로스는 실제 데이터의 분포와 모델이 예측한 분포라는 두 개의 서로 다른 확률 분포를 '교차'하여 비교한다는 의미이다.

따라서 크로스 엔트로피는 실제 값에 기반한 정보량(엔트로피)을, 모델이 예측한 값을 사용하여 계산함으로써, 실제 분포와 예측 분포가 얼마나 다른지(불일치 정도)를 측정한다는 의미이다.


# 크로스 엔트로피를 제외한 다른 손실함수

크로스 엔트로피 말고도 손실함수는 문제 유형, 모델 가정에 따라 꽤 다양하게 쓰인다.

## 분류 문제에서 쓰이는 손실함수

1. 힌지 손실 (Hinge Loss)
- 주로 SVM에서 사용
- 분류의 마진(margin)을 최대화 하는데 초점
<img width="323" height="54" alt="image" src="https://github.com/user-attachments/assets/f89081d8-8ada-482f-836d-901207f5f3de" />

특징
- 출력이 확률이 아님
- 잘 분류된 데이터라도 마진이 작으면 손실 발생
- 딥러닝보다는 전통적인 머신러닝에서 자주 사용

*SVM(Support Vector Machine): 분류와 회귀에 쓰이는 머신러닝 알고리즘, 데이터를 여유있게 나누는 경계를 찾는 알고리즘

2. 제곱 힌지 손실 (Squared Hinge Loss)
<img width="338" height="63" alt="image" src="https://github.com/user-attachments/assets/e37e7cac-99e6-49ba-98a7-2963b54e2df1" />

특징
- 힌지 손실보다 큰 오차에 더 강한 패널티
- 미분이 더 부드러움 -> 기울기가 부드럽게(점진적으로) 변화 -> 수렴 안정성이 높아짐

3. KL Divergence(발산) (Kullback–Leibler Divergence)

- 두 확률 분포 간 차이를 측정
<img width="442" height="102" alt="image" src="https://github.com/user-attachments/assets/17c64a1d-6d4f-40a9-a878-a2f4c4bca6f5" />

특징    
- 지식 증류(Knowledge Distillation)
- VAE에서 핵심적으로 사용
- 크로스 엔트로피와 밀접한 관계 있음 (Cross Entropy = Entropy + KL)

*지식 증류: 확률분포에 담긴 상대적 유사도 정보(지식)를 그대로 옮긴다.
*VAE(Variational AutoEncoder): 데이터가 만들어지는 확률 구조를 학습하는 오토인코더(모델)

## 회귀 문제에서 쓰이는 손실함수

1. 평균 제곱 오차(MSE)

<img width="250" height="84" alt="image" src="https://github.com/user-attachments/assets/185b812f-b64a-4575-9a8a-ffa6240876c4" />

특징
- 가장 기본적인 회귀 손실
- 큰 오차에 매우 민감

2. 평균 절대 오차 (MAE)
<img width="258" height="92" alt="image" src="https://github.com/user-attachments/assets/50ed42fb-02bb-4953-b839-c7c1c0ae47f1" />

특징
- 이상치에 강함
- 미분 불연속점 존재 (0에서)

3. 허버 손실 (Huber Loss)

<img width="449" height="117" alt="image" src="https://github.com/user-attachments/assets/a8cb6a16-1aa1-400f-ade3-a141cd136fb1" />

특징
- MSE + MAE의 절충
- 작은 오차엔 제곱, 큰 오차엔 절댓값
- 실제 딥러닝 회귀에서 많이 사용

.. _ppo2:

.. automodule:: stable_baselines3.ppo

PPO
===

근거리 정책 최적화 <https://arxiv.org/abs/1707.06347>`_ 알고리즘은 A2C(여러 작업자 보유)와 TRPO(신뢰 영역 사용)의 아이디어를 결합합니다.
와 TRPO(신뢰 영역을 사용하여 액터를 개선)의 아이디어를 결합한 것입니다.

주요 아이디어는 업데이트 후 새 정책이 이전 정책에서 너무 멀지 않아야 한다는 것입니다.
이를 위해 PPO는 클리핑을 사용하여 너무 큰 업데이트를 피합니다.


.. 참고::

  PPO에는 문서화되지 않은 원래 알고리즘의 몇 가지 수정 사항이 포함되어 있습니다.
  장점은 정규화되고 값 함수도 클리핑할 수 있습니다.


참고
-----

- 원본 논문: https://arxiv.org/abs/1707.06347
- Arxiv 인사이트 채널의 PPO에 대한 명확한 설명: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI 블로그 게시물: https://blog.openai.com/openai-baselines-ppo/
- 스피닝업 가이드: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- 37 구현 세부 정보 블로그: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/


사용할 수 있나요?
----------

.. 참고::

  PPO의 반복 버전은 기여 저장소(https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)에서 사용할 수 있습니다.

  그러나 더 간단하고 빠르며 일반적으로 경쟁력 있는 대안으로 간단한 프레임 스태킹으로 시작하는 것이 좋습니다.
  간단한 프레임 스태킹으로 시작하는 것을 권장합니다: https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity--VmlldzoxOTI4NjE4
  프록젠 논문 부록 그림 11도 참조하세요. <https://arxiv.org/abs/1912.01588>`_.
  실제로는 ``VecFrameStack``을 사용하여 여러 관측치를 스택할 수 있습니다.


- 반복 정책: ❌
- 다중 처리: ✔️
- 체육관 공간:


============= ====== ===========
우주 활동 관찰
============= ====== ===========
이산 ✔️ ✔️
박스 ✔️ ✔️
멀티디스크리트 ✔️ ✔️
MultiBinary ✔️ ✔️
Dict ❌ ✔️
============= ====== ===========

예제
-------

이 예제는 라이브러리와 그 기능의 사용법을 보여주기 위한 것으로, 학습된 에이전트가 해당 환경을 해결하지 못할 수도 있습니다. 최적화된 하이퍼파라미터는 RL Zoo `저장소 <https://github.com/DLR-RM/rl-baselines3-zoo>`_에서 찾을 수 있습니다.

4개의 환경을 사용하여 ``카트폴-v1``에서 PPO 에이전트를 훈련합니다.

.. 코드 블록:: 파이썬

  import gym

  안정된_베이스라인3에서 PPO를 가져옵니다.
  from stable_baselines3.common.env_util import make_vec_env

  # 병렬 환경
  env = make_vec_env("CartPole-v1", n_envs=4)

  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("ppo_cartpole")

  델 모델 # 제거하여 저장 및 로딩을 보여줍니다.

  model = PPO.load("ppo_cartpole")

  obs = env.reset()
  동안 True:
      action, _states = model.predict(obs)
      obs, 보상, 완료, 정보 = env.step(action)
      env.render()


결과
-------

아타리 게임
^^^^^^^^^^^

전체 학습 곡선은 '관련 PR #110 <https://github.com/DLR-RM/stable-baselines3/pull/110>`_'에서 확인할 수 있습니다.


PyBullet 환경
^^^^^^^^^^^^^^^^^^^^^

6개의 시드를 사용한 PyBullet 벤치마크(2백만 단계)의 결과입니다.
전체 학습 곡선은 '관련 이슈 #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_에서 확인할 수 있습니다.


.. 참고::

  gSDE 논문 <https://arxiv.org/abs/2005.05719>`_의 하이퍼파라미터가 사용되었습니다(파이불릿 환경에 맞게 튜닝되었기 때문에).


*가우시안*은 구조화되지 않은 가우시안 노이즈가 탐색에 사용됨을 의미합니다,
그렇지 않으면 *gSDE*(일반화된 상태 의존 탐색)가 사용됩니다.

+--------------+--------------+--------------+--------------+-------------+
| 환경 | A2C | A2C | PPO | PPO |
+==============+==============+==============+==============+=============+
| 가우시안 | gSDE | 가우시안 | gSDE |
+--------------+--------------+--------------+--------------+-------------+
| HalfCheetah | 2003 +/- 54 | 2032 +/- 122 | 1976 +/- 479 | 2826 +/- 45 |
+--------------+--------------+--------------+--------------+-------------+
| 개미 | 2286 +/- 72 | 2443 +/- 89 | 2364 +/- 120 | 2782 +/- 76 |
+--------------+--------------+--------------+--------------+-------------+
| 호퍼 | 1627 +/- 158 | 1561 +/- 220 | 1567 +/- 339 | 2512 +/- 21 |
+--------------+--------------+--------------+--------------+-------------+
| 워커2D | 577 +/- 65 | 839 +/- 56 | 1230 +/- 147 | 2019 +/- 64 |
+--------------+--------------+--------------+--------------+-------------+


결과를 어떻게 복제하나요?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rl-zoo 리포지토리 <https://github.com/DLR-RM/rl-baselines3-zoo>`_를 복제합니다:

.. 코드 블록:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


벤치마크를 실행합니다(``$ENV_ID``를 위에서 언급한 환경으로 바꿉니다):

.. 코드 블록:: bash

  python train.py --algo ppo --env $ENV_ID --eval-episodes 10 --eval-freq 10000


결과를 플롯합니다(여기서는 PyBullet 환경에만 해당):

.. 코드 블록:: bash

  파이썬 스크립트/all_plots.py -a

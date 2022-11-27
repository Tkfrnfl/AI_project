# AI_project
조별과제 프로젝트입니다.

1.pip install -r requirements.txt requirements 설치후,

2.csv데이터 파일에서 원하는 나라 이름 복사후, main.py 80줄에 이름 복붙

3.python main.py
메인 파일 실행

** adam>> sgd 로 변경을 원하는 경우 NN.py 112 adam부분 주석처리후 106줄 sgd 부분 주석 해제

**데이터 특성상 lr값이 작아질수록 loss값 수렴까지 큰 epoch(약500이상)가 필요하므로 주의!

**지역 극소값에 빠져서 loss가 줄어들지 않는 경우가 종종 발생합니다.(예측값 보면 터무니없은 값이 나옴) loss값이 줄어들지 않을땐 재실행하여 rand weight값 재설정 or lr 조정이 필요합니다.

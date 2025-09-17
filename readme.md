## 주요 파일 설명

main.py : 자동 생성 파일
train.py : 커스텀 객체 학습
test.py : 모델 테스트

## 실습 script

기초 설명 darknet, ultralytics
https://github.com/AlexeyAB/darknet
- darknet git
https://github.com/ultralytics/ultralytics
- ultralytics git

google slide git clone

### 일반 버전

``` bash
# 실행 후 코드 보며 설정 설명
python train.py
# 코드 보며 설명
# test.py:5 pt 파일 경로 설정
# test.py:8 장치 번호 바꾸기
python test.py

# label-studio 실습
pip install label-studio
label-studio run
```

### uv 버전

``` bash
uv sync
# 실행 후 코드 보며 설정 설명
uv run train.py
# 코드 보며 설명
# test.py:5 pt 파일 경로 설정
# test.py:8 장치 번호 바꾸기
uv run test.py
train.py

# label-studio 실습
uv run label-studio
```

## 더 해볼 수 있는 것

https://docs.ultralytics.com/datasets/detect/coco8/#introduction
- yaml 예시

https://docs.ultralytics.com/modes/train/#multi-gpu-training
- gpu 사용해서 yolo 훈련시키기

https://github.com/ultralytics/ultralytics
- ultralytics git 페이지
- 무엇을 할 수 있는지

## 프로젝트에 사용한 script

```
uv init
uv add opencv-python
uv add ultralytics
uv add label-studio
```

# 기계학습을 활용한 유물공간 온습도 예측관리 시스템

<p align="center">
  <br>
  <img src="./readme_img/wc.jpg">
  <br>
  <img src="./readme_img/wc1.jpg">
  <br>
</p>

<br>


## 프로젝트 소개


<br>


### 프로젝트 동기
<p align="justify">
유물수명에 가장 많은 영향을 끼치는 요인이 온도, 습도인데, 유물이나 목재, 종이 등을 보관 및 관리하는 공간에서 단순하게 온도, 습도를 테이블형태로만 가져오는 형식이나 수기로 작성하는 형식은 불편하고 비효율적이라고 생각이 들었습니다.<br>
따라서 유물 공간을 관리하는 사람의 입장에서 더 편리하고 효율적인 시스템을 개발하기 위해 프로젝트를 진행하게 되었습니다.
</p>


<br>

###  프로젝트 목적
- 온습도를 자동화 저장, 시각화를 통해 직관적인 분석 용이
- 당일의 온습도를 예측하여 예측 온습도를 가지고 위험대비 
- 위험 알림을 통하여 효율적인 관리
- 통합적으로 관리 비용 절감



<br>

## 기술 스택

| Html | Css | js | BootStrap |
| :--------: | :--------: | :------: | :-----: | 
|   ![html]    |   ![css]    | ![js] | ![bootstrap] |

| Flask | MognoDB | AWS EC2 | Api | Plotly | TensorFlow |
| :--------: | :--------: | :------: | :-----: | :-----: | :-----: | 
|   ![flask]    |   ![mongodb]    | ![ec2] | ![api] | ![plotly] | ![tensorflow] |


<br>

## 구현 기능

### 1.일정한 간격으로 온습도 데이터 저장
#### 스마트 센서의 Api와 파이썬 스케줄러를 이용하여 온습도 데이터를 10분 간격으로 저장(자동화)

  ```python
from apscheduler.schedulers.background import BackgroundScheduler
  
schedule_1 = BackgroundScheduler()

@schedule_1.scheduled_job('cron', hour='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23', 
                           minute='0,10,20,30,40,50', id='thdata_job') #10분간격으로 실행
def thdata_job():
    exec(open("Thdataapi.py", encoding='utf-8').read())  #api를 이용해 데이터를 가져오는 코드
    
```    


<br>


### 2.온습도 시각화 기능
#### 관리자가 직관적으로 온습도 데이터의 해석을 가능하게 하는 기능 (직관적인 분석)
- #### MongoDB에 저장되어 있는 데이터를 가지고 Plotly를 이용하여 시각화
<p align="center">
  <br>
  <img src="./readme_img/wc2.jpg">
  <br>
</p>

- #### 오늘 날짜 기준으로 업데이트 되고있는 데이터를 시각화하는 장면, 애니메이션을 이용해 동적인 효과 제공

<p align="center">
  <br>
  <img src="./readme_img/wc3.jpg">
  <br>
</p>


- #### 오늘 이전의 데이터들을 가지고 시각화하고, 슬라이더 기능을 이용하여 자세하게 관측 가능

<br>


### 3.온습도 예측 기능
####  과거에 데이터를 가지고 미래의 온습도를 예측한 결과를 제시하는 기능 (위험 대비)
- #### 미리 학습시킨 LSTM 모델에 오늘 날짜 기준 기상청 데이터를 인풋으로 예측데이터를 보여줌
<p align="center">
  <br>
  <img src="./readme_img/wc4.jpg">
  <br>
</p>

<br>

<br>


### 4.위험 알림 기능
#### 온도나 습도가 안전범위를 벗어났으면 알람을 주는 기능 (효율적인 온습도 관리)
- #### 왼쪽의 Now risk는 Js로 1분간격으로 Api를 받아서 현재 상태를 보여줌
- #### 오른쪽은 파이썬 스케줄러를 이용하여 30분 간격으로 온습도 데이터에 대한 위험도를 계산하여 표시됨
<p align="center">
  <br>
  <img src="./readme_img/wc5.jpg">
  <br>
 
</p>




<p align="justify">

</p>

<br>



<!-- Stack Icon Refernces -->

[html]: /readme_img/html.svg
[css]: /readme_img/css.svg
[bootstrap]: /readme_img/bootstrap.svg
[js]: /readme_img/js.svg
[flask]: /readme_img/flask.svg
[mongodb]: /readme_img/mongodb.svg
[ec2]: /readme_img/ec2.svg
[api]: /readme_img/api.svg
[plotly]: /readme_img/plotly.svg
[tensorflow]: /readme_img/tensorflow.svg


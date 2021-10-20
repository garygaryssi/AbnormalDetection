### Headlight Defective Part Detection

## Flowchart
![flowchart](https://user-images.githubusercontent.com/82126412/138033352-db7f2fcf-4bf5-4f45-a937-fad44dd04f9e.png)


## Data labeling
![img01](https://user-images.githubusercontent.com/82126412/138033417-12ad0ab5-99d9-421d-81f2-15bd4fe4dcc3.png)



## Model1 
![image](https://user-images.githubusercontent.com/82126412/138033602-1aba7af9-edcb-4634-acd9-08f8868b63de.png)
![image](https://user-images.githubusercontent.com/82126412/138033626-db46cb06-d9e5-4033-b642-abd8c8aebb5c.png)

-객체 검출 후 ROI 영역 저장

## Model2
![image](https://user-images.githubusercontent.com/82126412/138034100-be02de16-23e7-4628-b37c-2b9425a039db.png)

![image](https://user-images.githubusercontent.com/82126412/138034109-cf131ed7-a2cb-4454-bf4a-230315248cf4.png)

- Model1을 통한 ROI 영역내에서의 detection이 FP를 감소할 수 있다.

## Web server
![image](https://user-images.githubusercontent.com/82126412/138034614-4d5a594e-d96b-4e46-be1d-737c3774f340.png)
![image](https://user-images.githubusercontent.com/82126412/138034617-b6220938-706a-4138-b8b2-8bc212c943a5.png)
- Python Flask 프레임워크 사용
- 사용자로부터 분석 이미지를 입력 받기 위한 프론트 페이지 구성
- 학습 모델을 연동하여 결과를 돌려주는 백 엔드 구성

## Performance Evaluation
![성능평가](https://user-images.githubusercontent.com/82126412/138034743-d30aef96-50d2-4edd-9ac0-2e6705244ff6.png)

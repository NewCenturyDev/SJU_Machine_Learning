import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

# 맷플랏립 GUI 백엔드 설정
matplotlib.use('tkagg')

# Pandas 모든 열 보이게 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def load_mnist_data():
    # 손글씨 데이터 읽기
    print("===== 데이터 다운로드 시작 =====")
    mnist = fetch_openml('mnist_784', as_frame=False)
    print("===== 데이터 다운로드 완료 =====")

    print("===== 데이터에 포함된 키값 확인 =====")
    print(mnist.keys())

    return {
        'train': mnist['data'],
        'label': mnist['target']
    }


def visualize_mnist_data(train_data, train_label):
    # 자료구조 확인
    print("===== 데이터 차원 확인 =====")
    print("훈련 데이터 형태: {}".format(train_data.shape))
    print("훈련 라벨 형태: {}".format(train_label.shape))

    # 데이터 시각화 수행
    # 샘플 데이터 9개 표출
    # create figure with 3x3 subplots using matplotlib.pyplot
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    plt.gray()
    plt.subplots_adjust(hspace=0.35)

    # loop through subplots and add mnist images
    for i, ax in enumerate(axs.flat):
        ax.matshow(train_data[i].reshape(28, 28))
        ax.axis('off')
        ax.set_title('Number {}'.format(train_label[i]))

    # display the figure
    plt.show()


def forecast_cluster_labels(kmeans, actual_labels, verbose=True):
    # kmean 클러스터가 주어진 숫자 종류들(actual_labels) 중 가장 가까운 숫자의 라벨을 클러스터에 붙이도록 함

    probable_labels = {}

    if verbose:
        print("===== 각 클러스터로 분류된 데이터들의 추정 숫자 =====")
    for i in range(kmeans.n_clusters):

        # 클러스터의 라벨 인덱스를 확인
        labels = []
        index = np.where(kmeans.labels_ == i)

        # 각각의 클러스터에 속한 데이터들에게 실제 라벨을 매핑
        labels.append(actual_labels[index])

        # 가장 많이 출몰하는 라벨을 계산
        if len(labels[0]) == 1:
            cnt = np.bincount(labels[0])
        else:
            cnt = np.bincount(np.squeeze(labels))

        # 가장 많이 출몰하는 라벨로 클러스터의 probable label을 설정
        if np.argmax(cnt) in probable_labels:
            probable_labels[np.argmax(cnt)].append(i)
        else:
            probable_labels[np.argmax(cnt)] = [i]

        if verbose:
            print('클러스터 인덱스: {}, 예측 라벨(예상되는 숫자): {}'.format(i, np.argmax(cnt)))

    return probable_labels


def forecast_data_labels(x_labels, cluster_labels):
    # 클러스터 라벨을 추정된 라벨로 업데이트

    predicted_labels = np.zeros(len(x_labels)).astype(np.uint8)
    for i, cluster in enumerate(x_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


def do_train(train_data, train_label):
    # 검증 데이터와 학습 데이터 분리
    x_train, x_validation, y_train, y_validation = train_test_split(
        train_data, train_label, random_state=7
    )

    # 라벨값 string을 int numpy array로 변환
    y_train = np.array(list(map(int, y_train)))
    y_validation = np.array(list(map(int, y_validation)))

    # 1차원 어레이로 차원 감소
    x_train = x_train.reshape(len(x_train), -1)
    # 데이터를 0~1 사이로 노말라이징
    x_train = x_train.astype(float) / 255.

    print("===== 학습 데이터의 차원 확인 =====")
    print("학습 데이터: ".format(x_train.shape))
    print("학습 라벨: ".format(x_train[0].shape))

    # 라벨들을 검사하여 숫자(데이터 라벨) 종류의 개수를 확인
    n_digits = len(np.unique(y_validation))
    print("===== 라벨을 검사하여 숫자(데이터 라벨) 종류의 개수를 확인 =====")
    print(n_digits)

    # K-Means 클러스터링 모델 초기화 맟 헉숩 - 클러스터 개수 = 숫자 종류 개수
    kmeans = MiniBatchKMeans(n_clusters=n_digits, batch_size=100, compute_labels=True, init='k-means++',
                             max_iter=100, max_no_improvement=10,
                             n_init=3, random_state=7, reassignment_ratio=0.01, tol=0.0,
                             verbose=0)
    kmeans.fit(x_train)

    # 각 클러스터의 라벨 설정
    cluster_labels = forecast_cluster_labels(kmeans, y_train)
    x_clusters = kmeans.predict(x_train)
    predicted_labels = forecast_data_labels(x_clusters, cluster_labels)

    # 클러스터 라벨과 실제 라벨의 비교
    print("===== 클러스터 라벨과 실제 라벨의 비교 (표본: 20개) =====")
    print("예측한 라벨 (클러스터 라벨): {}".format(predicted_labels[:20]))
    print("실제 라벨: {}".format(y_train[:20]))

    # 클러스터 개수를 다르게 해서 테스트
    clusters = [10, 16, 32, 64, 128, 256]
    accuracy = []
    tries_cnt = 0
    for n_clusters in clusters:
        estimator = MiniBatchKMeans(n_clusters=n_clusters)
        estimator.fit(x_train)

        # 클러스터 성능 지표 표출
        print("===== 클러스터 성능 지표 확인 ({}개 클러스터로 분할한 경우) =====".format(clusters[tries_cnt]))
        print('클러스터의 개수: {}'.format(estimator.n_clusters))
        print('클러스터 응집도: {}'.format(estimator.inertia_))
        print('클러스터 구분성: {}'.format(metrics.homogeneity_score(y_train, estimator.labels_)))
        tries_cnt += 1

        # determine predicted labels
        cluster_labels = forecast_cluster_labels(estimator, y_train)
        predicted_number_label = forecast_data_labels(estimator.labels_, cluster_labels)

        # calculate and print accuracy
        print('예측 Accuracy: {}\n'.format(metrics.accuracy_score(y_train, predicted_number_label)))
        accuracy.append(metrics.accuracy_score(y_train, predicted_number_label))

    results = pd.DataFrame({
        'cluster_n': clusters,
        'accuracy': accuracy
    })
    print("===== 클러스터 개수에 따른 최종 성능 지표 =====")
    print(results)

    # 256일떄 가장 성능이 좋았음으로 256 채택
    # 검증 데이터의 차원 줄이기 및 정규화
    x_validation = x_validation.reshape(len(x_validation), -1)
    x_validation = x_validation.astype(float) / 255.

    # K-Means 클러스터링 모델 초기화 맟 학습 - 클러스터 개수 = 256
    kmeans = MiniBatchKMeans(n_clusters=256)
    kmeans.fit(x_train)
    cluster_labels = forecast_cluster_labels(kmeans, y_train, False)

    return {
        "model": kmeans,
        "labels": cluster_labels,
        "x_validation": x_validation,
        "y_validation": y_validation,
    }


def do_validation(kmeans, cluster_labels, x_validation, y_validation):
    # 검증 데이터 라벨 예측
    # test_clusters = kmeans.predict(x_validation)
    predicted_labels = forecast_data_labels(kmeans.predict(x_validation), cluster_labels)

    # 정확도 표기
    print("===== 최종 선택된 클러스터의 성능 지표 확인 (256개 클러스터로 분할한 경우) =====")
    print('예측 Accuarcy: {}\n'.format(metrics.accuracy_score(y_validation, predicted_labels)))


def do_visualize_sample36(x_train, y_train):
    # 1차원 어레이로 차원 감소
    x_train = x_train.reshape(len(x_train), -1)
    # 데이터를 0~1 사이로 노말라이징
    x_train = x_train.astype(float) / 255.
    # 라벨값 string을 int numpy array로 변환
    y_train = np.array(list(map(int, y_train)))

    # Initialize and fit KMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=36)
    kmeans.fit(x_train)

    # record centroid values
    centroids = kmeans.cluster_centers_

    # reshape centroids into images
    images = centroids.reshape(36, 28, 28)
    images *= 255
    images = images.astype(np.uint8)

    # determine cluster labels
    cluster_labels = forecast_cluster_labels(kmeans, y_train)

    # create figure with subplots using matplotlib.pyplot
    fig, axs = plt.subplots(6, 6, figsize=(20, 20))
    plt.gray()

    # loop through subplots and add centroid images
    # 36개의 클러스터를 쪼갰을 때, 각 클러스터의 중심 센트로이드 이미지 표출
    print("===== 센트로이드 이미지 표출 (36클러스터일때) =====")
    for i, ax in enumerate(axs.flat):

        # determine inferred label using cluster_labels dictionary
        for key, value in cluster_labels.items():
            if i in value:
                ax.set_title('Predicted Label: {}'.format(key))

        # add image to subplot
        ax.matshow(images[i])
        ax.axis('off')

    # display the figure
    fig.show()


# 메인 로직
data = load_mnist_data()
# 예시 데이터 보기
visualize_mnist_data(data["train"], data["label"])
train_result = do_train(data['train'], data['label'])
do_validation(train_result["model"], train_result["labels"], train_result["x_validation"], train_result["y_validation"])
do_visualize_sample36(data["train"], data["label"])

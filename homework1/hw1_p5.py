# P5 유클리시안 거리를 계산하기

import math


def calc_euclidean_dist(v1, v2):
    # 두 벡터 간의 유클리디안 유사도를 계산하는 함수
    # 4차원 좌표를 가진 벡터임으로 각 차원에 대해 반복
    # 거리의 제곱값
    dist_power_2 = 0
    # 각 차원의 v2 벡터의 i차원 좌표값의 제곱값에서 v1 벡터의 i좌표 좌표값을 빼서 차이를 더하는 것을 반복 (총 4개 차원)
    # 시그마 i=1부터 n=4까지 (p_i - q_i)^2에 대응함
    for i in range(4):
        dist_power_2 += (v2[i] - v1[i])**2
    # 거리의 제곱값의 제곱근을 산출
    # 루트(시그마 i=1부터 n=4까지 (p_i - q_i)^2)를 수행하는 것에 대응함
    result = math.sqrt(dist_power_2)
    return result


def main():
    # 메인 함수
    print("시작")
    # 데이터 정의
    data = [
        [1, 2, 3, 4],
        [1, 8, 1, 8],
        [10, 6, 4, 2],
        [0, 2, 4, 6]
    ]

    # 벡터간 유클리시안 거리 행렬 출력
    result_matrix = []
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            row.append(calc_euclidean_dist(data[i], data[j]))
        result_matrix.append(row)
    print(result_matrix)

    # 개발 수치 출력
    # 자기 자신과의 거리는 0임이 자명함으로 생략함
    # 포인터 i, j를 가지고 단순 이중 포문(중복값 포함) 또는
    # 내부 for문(j)의 종료조건을 가변으로 설정하는 이중포문(중복값 제외)으로 구현할 수도 있으나 명확히 하기 위해 코드를 나열함
    # 1번째 데이터의 벡터 v1(data[0])과 2번째 데이터의 벡터 v2(data[1])간의 거리 산출 후 출력
    print('1번째 데이터의 벡터 v1(data[0])과 2번째 데이터의 벡터 v2(data[1])간의 거리')
    print(calc_euclidean_dist(data[0], data[1]))
    # 1번째 데이터의 벡터 v1(data[0])과 3번째 데이터의 벡터 v3(data[2])간의 거리 산출 후 출력
    print('1번째 데이터의 벡터 v1(data[0])과 3번째 데이터의 벡터 v3(data[2])간의 거리')
    print(calc_euclidean_dist(data[0], data[2]))
    # 1번째 데이터의 벡터 v1(data[0])과 4번째 데이터의 벡터 v4(data[3])간의 거리 산출 후 출력
    print('1번째 데이터의 벡터 v1(data[0])과 4번째 데이터의 벡터 v4(data[3])간의 거리')
    print(calc_euclidean_dist(data[0], data[3]))
    # 2번째 데이터의 벡터 v2(data[1])과 3번째 데이터의 벡터 v3(data[2])간의 거리 산출 후 출력
    print('2번째 데이터의 벡터 v2(data[1])과 3번째 데이터의 벡터 v3(data[2])간의 거리')
    print(calc_euclidean_dist(data[1], data[2]))
    # 2번째 데이터의 벡터 v2(data[1])과 4번째 데이터의 벡터 v4(data[3])간의 거리 산출 후 출력
    print('2번째 데이터의 벡터 v2(data[1])과 4번째 데이터의 벡터 v4(data[3])간의 거리')
    print(calc_euclidean_dist(data[1], data[3]))
    # 3번째 데이터의 벡터 v3(data[2])과 4번째 데이터의 벡터 v4(data[3])간의 거리 산출 후 출력
    print('3번째 데이터의 벡터 v3(data[2])과 4번째 데이터의 벡터 v4(data[3])간의 거리')
    print(calc_euclidean_dist(data[2], data[3]))


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다. - 파일 실행시 main함수가 자동 실행되게 함
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조

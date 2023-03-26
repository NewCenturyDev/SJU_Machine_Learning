# P4 텍스트의 코사인 유사도를 구해 보기

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def vectorise_text(sentenses):
    # 단어 카운트 벡터라이저 생성 - 기본값은 1글자 단어를 무시하게 되어 있음으로 커스텀 토크나이저 정의(문장을 띄어쓰기 단위로 자르기만 함)
    vectorizer = CountVectorizer(tokenizer=lambda sentense: sentense.split(' '))
    # 문장 벡터화 진행 - fit_transform 함수를 호출하여 내부적으로 문장을 구성하는 단어 사전을 생성
    # 두 문장을 넣었음으로, 문장별로 등장하는 단어의 등장 횟수를 카운팅하여 배열 형태로 출력
    count_diff_matrix = vectorizer.fit_transform(sentenses)
    # 단어 사전 출력
    print('두 문장을 구성하는 단어의 종류 = 단어 사전 = 단어 등장 빈도 행렬 테이블의 X축(가로축)')
    print(vectorizer.get_feature_names_out())
    # 벡터화된 행렬 출력 양식: "(문장 인덱스, 단어 인덱스) 등장빈도"
    # 특정 문장에 등장하지 않는 단어는 표시되지 않음
    # 예를 들어 1번째 문장(인덱스 0)에서 단어사전의 0번쨰 단어('a')가 등장하는 빈도는 2회이고, 이것은 "(0, 0) 2"와 같이 표현됨.
    # 앞선 튜플은 15가지의 단어로 구성된 두 문장의 행렬 좌표(y행 - 세로, x열 - 가로)이고, 뒤의 정수값(등장 빈도)는 각 행렬 좌표의 요소 값임.
    # 특정 문장에 등장하지 않는 단어(=행렬의 요소값이 0인 좌표)는 생략되어 표시되지 않음.
    print('두 문장을 구성하는 단어 등장 빈도 행렬 출력 = "(y행, x열) 행렬값 = (문장 인덱스, 단어사전에서의 단어 인덱스) 등장빈도"')
    print('행렬의 요소값이 0인 요소(=특정 문장에 등장하지 않는 단어)는 표출되지 않고 생략됩니다')
    print(count_diff_matrix)
    return count_diff_matrix


def calc_cosine_simularity(count_matrix):
    # 코사인 유사도 측정
    print("코사인 유사도를 측정하여 두 문장간의 유사도를 산출")
    # 코사인 유사도 측정 함수를 호출 - 내부적으로 cos(d1,d2)=(d1*d2/||d1|| ||d2||)를 계산
    simularity = cosine_similarity(count_matrix[0:1], count_matrix[1:2])
    print(simularity)


def main():
    # 메인 함수
    print("시작")
    # 문장 정의
    sentenses = (
        "try not to become a man of success but rather try to become a man of value",
        "theories should be as simple as possible, but not simpler"
    )
    # 문장을 벡터화해서 각 문장을 구성하는 단어들의 등장 횟수에 대한 벡터(행렬) 구하기
    count_matrix = vectorise_text(sentenses)
    # 벡터화된 행렬을 기반으로 코사인 유사도 계산
    calc_cosine_simularity(count_matrix)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다. - 파일 실행시 main함수가 자동 실행되게 함
if __name__ == '__main__':
    main()

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조

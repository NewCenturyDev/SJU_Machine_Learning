import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

DATA_IN_PATH = './data/etf_data.csv'


def load_data():
    df = pd.read_csv(DATA_IN_PATH, low_memory=False, thousands=',')
    df.head()

    # 2010~2020 4월까지의 일별 국내상장 ETF 순매수 금액 데이터
    print('2010~2020 4월까지의 일별 국내상장 ETF 순매수 금액 데이터')
    df_copy = df.copy()
    print(df_copy.head())
    return df_copy


def encode_data(df_copy):
    # 종목명 가져오기 및 Symbol Name (거래일자) 제거
    col_name = df_copy.columns
    col_name.drop('Symbol Name')
    print('종목명')
    print(col_name)

    # 순매수 금액이 넘으면 1로, 아니면 0으로 인코딩
    print('순매수 금액이 넘으면 1로, 아니면 0으로 인코딩')
    for i in col_name:
        m_buy = int(df_copy[i]) > 100000
        m_none = int(df_copy[i]) < 100000

        df_copy.loc[m_buy, i] = 1
        df_copy.loc[m_none, i] = 0
    df_copy.drop(['Symbol Name'], inplace=True, axis=1)
    df_copy.head()

    # int 0, 1로 할당되어 있는 것을 True, False 형태로 변환
    df_copy_temp = df_copy == 1
    df_copy_temp.head()

    # 지지도가 0.1 이상인 것만 추림
    print('지지도가 0.1 이상인 것만 추림')
    itemset = apriori(df_copy_temp, min_support=0.1, use_colnames=True)
    itemset.sort_values(['support'], asending=False).head(10)

    # 지지도가 0.1 이상인 것 중에서 신뢰도가 0.1 이상인 것만 추림
    print('지지도가 0.1 이상인 것 중에서 신뢰도가 0.1 이상인 것만 추림')
    rules_ca = association_rules(itemset, metric='confidence', min_threshold=0.1)
    rules_ca = rules_ca[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules_ca.reset_index(inplace=True)
    rules_ca.head(10)
    rules_ca['antecedents'] = rules_ca['antecedents'].astype('str')
    rules_ca['consequents'] = rules_ca['consequents'].astype('str')
    rules_ca['antecedents'] = rules_ca['antecedents'].str.replace("frozenset", "")
    rules_ca['consequents'] = rules_ca['consequents'].str.replace("frozenset", "")
    rules_ca.antecedents = rules_ca.antecedents.apply(''.join).str \
        .replace('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
    rules_ca.consequents = rules_ca.consequents.apply(''.join).str. \
        replace('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
    rules_ca['ant_con'] = rules_ca[['antecedents', 'consequents']].apply(lambda x: ' '.join(x), axis=1)

    return {
        'ca_support': rules_ca.as_matrix(columns=['support']),
        'ca_confidence': rules_ca.as_matrix(columns=['confidence']),
        'ca_lift': rules_ca.as_matrix(columns=['lift'])
    }


def show_graph(ca_support, ca_confidence, ca_lift):
    plt.figure(figsize=(15, 3))

    for i in range(len(ca_support)):
        ca_support[i] = ca_support[i]
        ca_confidence[i] = ca_confidence[i]

    plt.subplot(161)
    plt.title("support & confidence")
    plt.scatter(ca_support, ca_confidence, alpha=0.5, marker="*")
    plt.xlabel('support')
    plt.ylabel('confidence')
    # plt.show()

    for i in range(len(ca_support)):
        ca_lift[i] = ca_lift[i]
        ca_confidence[i] = ca_confidence[i]

    plt.subplot(163)
    plt.title("lift & confidence")
    plt.scatter(ca_lift, ca_confidence, alpha=0.5, marker="*")
    plt.xlabel('lift')
    plt.ylabel('confidence')
    # plt.show()

    for i in range(len(ca_support)):
        ca_support[i] = ca_support[i]
        ca_confidence[i] = ca_confidence[i]

    plt.subplot(165)
    plt.title("support & lift")
    plt.scatter(ca_support, ca_lift, alpha=0.5, marker="*")
    plt.xlabel('support')
    plt.ylabel('lift')

    plt.show()


raw_data = load_data()
data = encode_data(raw_data)
show_graph(data['ca_support'], data['ca_confidence'], data['ca_lift'])




1. 텍스트 데이터 정규화
- 토큰화 ( Tokenization ) 
문서를 문장 단위로, 문장을 단어 단위로 쪼개는 것을 말한다. 주로 사용되는 방법으로 N-gram 방법이 있는데 N-gram에 대한 개념은 여기를 참고하자.
- 필터링
- 불용어 제거 
말 그대로 불필요한 단어 즉, 텍스트로부터 주요한 정보를 얻는 것에 영향을 미치지 않는 단어들을 제거한다. 영어의 불용어에 대한 예시로는 'he', 'is', 'will' 등이 되겠다.
- 오타 수정
- 어근(단어의 원형 추출)

2. BOW ( Bag-of-Word )
텍스트를 숫자로 변형해주는 과정을 Feature Vecorizer(피처 벡터화)라고 한다. 피처 즉, 텍스트를 벡터화시키는 과정에서 크게 두 가지 방법이 존재한다. 단순히 여러개의 문서에 발생하는 단어의 빈도수에 기반한 CountVectorizer 방법, 이에 반해 "여러개의 문서 모두 다 많이 등장하는 단어는 정보를 추출하는 데에 별로 영향을 주지 않을거야"라는 사고에 기반한 Tf-idf(Term frequency - Inverse document frequency) Vectorizer 방법이 있다. 


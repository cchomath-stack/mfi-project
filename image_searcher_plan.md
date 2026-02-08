# 기획서: 이미지 기반 유사 문항 검색기 (Image Searcher)

## 1. 개요
새로 생성된 `mcat2.problem_image_embeddings` 데이터를 활용하여, 사용자가 업로드한 이미지와 가장 유사한 기존 문항 5개를 찾아내고 그 출처(Source)를 보여주는 검색 시스템입니다.

## 2. 검색 로직 프로세스
1. **입력**: 검색하고자 하는 문제 이미지 하나.
(ctrl+c -> ctrl+v 로도 이미지 바로 입력 가능하도록)
2. **변환**: 동일한 모델(`openai/clip-vit-base-patch32`)을 사용하여 입력 이미지를 **벡터(Embedding)**로 변환.
3. **매칭**: `mcat2.problem_image_embeddings` 테이블 내의 모든 벡터와 **코사인 유사도(Cosine Similarity)**를 계산.
4. **추출**: 유사도가 가장 높은 상위 5개 `problem_id` 추출.
5. **결과 조회**: 추출된 ID를 기반으로 문항의 이미지, 텍스트, 그리고 **가장 중요한 '출처(Source)' 정보**를 Join하여 출력.

## 3. 데이터베이스 연결 (Join 구조)

문항의 출처를 정확히 밝히기 위해 아래 테이블들을 결합합니다.

### 관련 테이블 및 컬럼
- **`mcat2.problem_image_embeddings`**: 검색 대상 벡터 데이터 (ID 매칭용)
- **`mcat2.problems_problem`**: 문항 이미지 URL 및 텍스트 정보
- **`mcat2.problems_source`**: 실제 출처 명칭 (예: "2024년 3월 학평", "수학의 정석" 등)
- **`mcat2.problems_problem_with_source`**: 문항(`problem`)과 출처(`source`)를 이어주는 중간 테이블

### SQL Join 예시
```sql
SELECT 
    p.problem_image_url, 
    s.title AS source_title, 
    s.exam_year, 
    s.exam_month
FROM mcat2.problems_problem p
JOIN mcat2.problems_problem_with_source pws ON p.id = pws.problem_id
JOIN mcat2.problems_source s ON pws.source_id = s.id
WHERE p.id = '찾아낸_UUID';
```

---

## 4. AI 에이전트(Antigravity)에게 내릴 구체적 명령 (Prompt)

새로운 대화창에서 아래 내용을 그대로 복사해서 전달하시면 구현을 시작할 것입니다.

> **[명령 전문]**
>
> "수학 문제 이미지 유사도 검색기"를 파이썬으로 구현해줘.
>
> **1. 환경 정보 (DB 주소):**
> - POSTGRES: `postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki`
>
> **2. 구현 요구사항:**
> - **이미지 벡터화**: `openai/clip-vit-base-patch32` 모델을 사용하여 입력 이미지를 벡터로 변환할 것.
> - **유사도 검색**: `mcat2.problem_image_embeddings` 테이블의 데이터와 '코사인 유사도'를 계산하여 상위 5개를 추출할 것.
> - **출처 매핑**: 찾아낸 문항 ID를 기반으로 `mcat2.problems_problem`, `mcat2.problems_problem_with_source`, `mcat2.problems_source` 테이블을 Join하여 **문항 이미지 URL**과 **출처 제목(title)**을 가져올 것.
> - **인터페이스**: 이미지를 입력받으면 결과(이미지 URL, 유사도 점수, 출처 이름) 5개를 리스트로 출력하는 간단한 스크립트를 만들어줘.
>
> **3. 주의사항:**
> - `problem_id`는 UUID 형식이므로 타입 에러가 나지 않게 처리할 것.
> - `pgvector` 확장이 설치되어 있지 않을 수 있으므로, 파이썬(numpy/torch) 단에서 유사도 계산 로직을 포함할 것. (데이터가 4만 건 수준이므로 파이썬 처리 가능)

---

## 5. 필요한 주소 총정리 (메모용)
- **Database**: `postgresql://db_member4:csm17csm17!@43.201.182.105:5432/tki`
- **Model**: `openai/clip-vit-base-patch32`
- **Schema**: `mcat2`
- **Target Table**: `problem_image_embeddings`

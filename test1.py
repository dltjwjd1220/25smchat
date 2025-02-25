import pandas as pd
import ollama
import re
from datetime import datetime

class AcademicCalendarRAG:
    def __init__(self, csv_path='학사일정.csv'):
        self.df = self._load_data(csv_path)

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        df['Year'] = df['Start'].dt.year

        # 학기 구분 (1학기 / 2학기)
        df['Semester'] = df['Title'].apply(self._extract_semester)

        # "수강신청", "장바구니 수강신청", "수강신청 정정 및 취소" 구분 추가
        df['Category'] = df['Title'].apply(self._extract_category)

        df['document'] = df.apply(lambda row:
                                  f"{row['Title']} 일정은 {row['Start'].strftime('%Y년 %m월 %d일')}부터 "
                                  f"{row['End'].strftime('%Y년 %m월 %d일')}까지입니다.", axis=1)
        return df

    def _extract_semester(self, title):
        """ 학기 구분 (1학기 / 2학기) """
        if '1학기' in title:
            return '1학기'
        elif '2학기' in title:
            return '2학기'
        return None  # 학기 구분이 없는 일정

    def _extract_category(self, title):
        """ '수강신청', '장바구니 수강신청', '수강신청 정정 및 취소' 구분 """
        if '장바구니' in title:
            return '장바구니'
        elif '수강신청 정정' in title or '수강신청 취소' in title:
            return '정정'
        elif '수강신청' in title:
            return '수강신청'
        return None  # 기타 일정

    def _extract_year_and_semester(self, question):
        """ 질문에서 연도와 학기를 추출 (없으면 현재 연도 반환) """
        match = re.search(r'(\d{4})', question)
        year = int(match.group(1)) if match else datetime.now().year

        if '1학기' in question:
            semester = '1학기'
        elif '2학기' in question:
            semester = '2학기'
        else:
            semester = None  # 학기 정보 없음

        # 연도를 포함한 질문에서 연도 제거하여 키워드 추출
        clean_question = re.sub(r'\d{4}', '', question).strip()
        
        return year, semester, clean_question

    def _get_relevant_documents(self, query, year, semester=None):
        """ 특정 연도와 학기의 학사일정 검색 (정확한 키워드 매칭) """
        filtered_df = self.df[self.df['Year'] == year]

        if semester:
            filtered_df = filtered_df[filtered_df['Semester'] == semester]

        # "수강신청" 질문에서는 "수강신청 정정 및 취소" 일정 제외
        if '수강신청' in query and '정정' not in query and '취소' not in query:
            filtered_df = filtered_df[filtered_df['Category'] == '수강신청']

        # "장바구니 수강신청" 질문에서는 장바구니 일정만 검색
        elif '장바구니' in query:
            filtered_df = filtered_df[filtered_df['Category'] == '장바구니']

        # 날짜순 정렬
        filtered_df = filtered_df.sort_values(by='Start')

        # 결과가 없으면 빈 리스트 반환
        if filtered_df.empty:
            return []

        return [doc['document'] for _, doc in filtered_df.iterrows()]

    def get_answer(self, question):
        """ LLM을 사용해 답변 생성 """
        year, semester, clean_question = self._extract_year_and_semester(question)
        relevant_contexts = self._get_relevant_documents(clean_question, year, semester)

        if not relevant_contexts:
            return f"{year}년의 관련 학사일정 정보를 찾을 수 없습니다."

        formatted_context = "\n".join(relevant_contexts)

        prompt = f"""당신은 상명대학교 학생들을 위한 챗봇입니다. 다음 규칙을 바탕으로 답변해 주세요.
1. 질문을 그대로 반복하지 말고, 명확하고 간결하게 일정 정보를 제공할 것.
2. "수강신청", "장바구니 수강신청", "수강신청 정정 및 취소"를 구분하여 답변할 것.
3. 질문에 특정 학기(1학기 또는 2학기)가 포함되어 있으면 해당 학기의 일정만 출력할 것.
4. 질문에 학기가 없으면 1학기 일정과 2학기 일정을 각각 정리해서 답변할 것.
5. 관련된 일정이 여러 개일 경우 날짜 순으로 정렬하여 제공할 것.
6. "~는 ~부터 ~입니다." 또는 "~는 ~입니다." 형식으로만 답변할 것.
7. 관련 일정이 없으면 "관련 정보를 찾을 수 없습니다."라고 답변할 것.

현재 연도: {year}
관련 학사일정 정보:
{formatted_context}

질문: {question}
위 정보를 바탕으로 질문에 답변하세요.
"""

        try:
            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


def main():
    rag_system = AcademicCalendarRAG()
    print("학사일정 RAG Load 완료.")
    print("종료하려면 'quit' 또는 'exit'를 입력.")

    while True:
        question = input("\n질문: ")
        if question.lower() in ['quit', 'exit']:
            break

        answer = rag_system.get_answer(question)
        print("\n답변:", answer)


if __name__ == "__main__":
    main()

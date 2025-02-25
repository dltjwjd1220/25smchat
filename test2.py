import pandas as pd
import json
import re
from datetime import datetime
from synonym_manager import SynonymManager  # 동의어 사전 불러오기

# 학사일정 데이터 로드
file_path = "hagsailjeong.csv"  # 학사일정 CSV 파일 경로
df = pd.read_csv(file_path)

# 날짜 형식 변환
df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
df["End"] = pd.to_datetime(df["End"], errors="coerce")

# 연도 및 학기 정보 추출
df["Year"] = df["Start"].dt.year
df["Semester"] = df["Title"].apply(lambda x: "1학기" if "1학기" in x else ("2학기" if "2학기" in x else None))

# 키워드 필터링
df["Keyword"] = df["Title"].apply(lambda x: "수강신청" if "수강신청" in x else ("성적입력" if "성적입력" in x else None))

class AcademicCalendarRAG:
    def __init__(self, df, synonym_path='synonyms.json'):
        self.df = df  # 학사일정 데이터프레임
        self.synonym_manager = SynonymManager(synonym_path)

    def _extract_year_and_semester(self, question):
        """ 질문에서 연도와 학기를 추출 (없으면 전체 연도 조회) """
        question = question.lower().strip()
        match = re.search(r'(\d{4})', question)
        year = int(match.group(1)) if match else None  # 연도가 없으면 None 처리

        semester = None
        if '1학기' in question:
            semester = '1학기'
        elif '2학기' in question:
            semester = '2학기'

        # 연도를 포함한 질문에서 연도 제거하여 키워드 추출
        clean_question = re.sub(r'\d{4}', '', question).strip()

        # 동의어 변환
        normalized_query = self.synonym_manager.get_standard_term(clean_question)
        return year, semester, normalized_query

    def _get_relevant_documents(self, query, year=None, semester=None):
        """ 특정 연도와 학기의 학사일정 검색 (연도 없음 = 전체 연도 검색) """
        filtered_df = self.df

        # 연도가 주어진 경우 해당 연도가 포함된 Title만 검색
        if year:
            filtered_df = filtered_df[filtered_df['Title'].str.contains(str(year), na=False, case=False)]

        # 학기가 주어진 경우 해당 학기의 일정만 검색
        if semester:
            filtered_df = filtered_df[filtered_df['Semester'] == semester]

        # 키워드 검색 (Title, Keyword 컬럼에서 검색)
        filtered_df = filtered_df[
            filtered_df['Title'].str.contains(query, na=False, case=False) |
            filtered_df['Keyword'].str.contains(query, na=False, case=False)
        ]

        # 날짜순 정렬
        filtered_df = filtered_df.sort_values(by='Start')

        if filtered_df.empty:
            return []

        return [f"{doc['Title']} 일정은 {doc['Start'].strftime('%Y년 %m월 %d일')}부터 "
                f"{doc['End'].strftime('%Y년 %m월 %d일')}까지입니다."
                for _, doc in filtered_df.iterrows()]

    def get_answer(self, question):
        """ 검색 결과를 반환 """
        year, semester, clean_question = self._extract_year_and_semester(question)
        relevant_contexts = self._get_relevant_documents(clean_question, year, semester)

        if not relevant_contexts:
            return f"{year if year else '모든'}년의 관련 학사일정 정보를 찾을 수 없습니다."

        return "\n".join(relevant_contexts)


# RAG 시스템 초기화
rag_system = AcademicCalendarRAG(df)

# ✅ **사용자가 직접 질문할 때만 실행되도록 변경**
def main():
    print("🔹 상명대학교 학사일정 챗봇이 실행되었습니다.")
    print("💡 질문을 입력하세요. (예: '2025년 1학기 수강신청 언제야?')")
    print("🛑 종료하려면 'exit' 또는 'quit'을 입력하세요.")

    while True:
        question = input("\n질문: ")

        if question.lower() in ['exit', 'quit']:
            print("🛑 챗봇을 종료합니다.")
            break

        answer = rag_system.get_answer(question)
        print("\n답변:", answer)


# ✅ 프로그램 실행 (사용자 입력 대기)
if __name__ == "__main__":
    main()

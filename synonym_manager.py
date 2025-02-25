import json
import re
from pathlib import Path


class SynonymManager:
    def __init__(self, csv_path='hagsailjeong.csv', synonym_path='synonyms.json'):
        self.synonym_path = Path(synonym_path)
        self.synonyms = self._initialize_synonyms()

    def _initialize_synonyms(self):
        """ 기존 동의어 사전 로드 또는 새로 생성 """
        if self.synonym_path.exists():
            return self._load_synonyms()
        return {}

    def _load_synonyms(self):
        """ 동의어 사전 로드 """
        with open(self.synonym_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_standard_term(self, query):
        """ 
        사용자의 질문을 동의어 사전에 매칭하여 표준화된 키워드 반환 
        - 기존에는 query가 동의어 사전의 키워드와 완전히 일치해야 변환되었지만,
          이제는 query에 동의어가 포함되어 있으면 자동 변환하도록 개선함.
        """
        query = query.lower().strip()

        for standard_term, synonyms in self.synonyms.items():
            for synonym in synonyms:
                if synonym in query:  # query에 동의어가 포함되어 있다면 변환
                    return standard_term

        return query  # 동의어 매칭 실패 시 원래 질문 반환

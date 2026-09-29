"""SQLite DB 연결/모델 계층 (docs/feature_spec.md 4.4 스키마 초안 참고).

- database.py: 공용 SQLite 연결 설정 (테이블 코드는 여기 연결만 가져다 쓴다)
- artifacts.py: artifacts 테이블 (구현 완료)
- conversations.py: conversations / messages 테이블 - 채팅 담당자가 이 파일로 추가
"""

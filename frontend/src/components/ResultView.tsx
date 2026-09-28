import { FormEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Button, Mascot, ScanIcon, SendIcon, artifactPhoto } from "./common";

function AssistantMessage({
  children,
  source,
  waiting = false,
}: {
  children?: ReactNode;
  source?: boolean;
  waiting?: boolean;
}) {
  return (
    <div className="message-row assistant-row">
      <Mascot small />
      <div className="message-stack">
        <div className={`bubble assistant-bubble ${waiting ? "thinking" : ""}`}>
          {waiting ? (
            <>
              <span>민속이가 답을 생각하고 있어요</span>
              <span className="loading-dots dark-dots" aria-label="로딩 중">
                <i />
                <i />
                <i />
              </span>
            </>
          ) : (
            children
          )}
        </div>
        {source && <span className="source">출처: 국립중앙박물관</span>}
      </div>
    </div>
  );
}

function UserMessage({ children }: { children: ReactNode }) {
  return (
    <div className="message-row user-row">
      <div className="bubble user-bubble">{children}</div>
    </div>
  );
}

export function ResultView({ onRescan }: { onRescan: () => void }) {
  const [messages, setMessages] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [thinking, setThinking] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setThinking(false), 2400);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, thinking]);

  const submitQuestion = (question: string) => {
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;
    setMessages((current) => [...current, cleanQuestion]);
    setText("");
    setThinking(true);
    window.setTimeout(() => setThinking(false), 1500);
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submitQuestion(text);
  };

  return (
    <div className="result-view">
      <aside className="artifact-panel">
        <div className="frozen-photo">
          <img src={artifactPhoto} alt="인식된 금동 반가사유상" />
          <span className="freeze-badge">인식 완료</span>
        </div>
        <div className="artifact-info">
          <span className="artifact-label">국보 · 삼국시대</span>
          <p className="artifact-name">금동 반가사유상</p>
          <div className="match-row">
            <span>AI 일치도</span>
            <strong>94%</strong>
          </div>
          <div
            className="progress-track"
            role="progressbar"
            aria-valuenow={94}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="AI 일치도 94%"
          >
            <span />
          </div>
          <p className="artifact-caption">
            한쪽 다리를 다른 쪽 무릎에 올리고 깊은 생각에 잠긴 보살상이에요.
          </p>
        </div>
        <Button className="secondary-button" onClick={onRescan}>
          <ScanIcon />
          다시 비추기
        </Button>
      </aside>

      <section className="chat-panel">
        <div className="chat-heading">
          <div>
            <span className="section-kicker">민속이와 더 알아보기</span>
            <p>금동 반가사유상 이야기</p>
          </div>
          <span className="online-label">
            <i />
            AI 도슨트
          </span>
        </div>

        <div className="conversation">
          <AssistantMessage source>
            이 유물은 <strong>금동 반가사유상</strong>이에요. 오른발을 왼쪽
            무릎에 올리고 손가락을 뺨에 댄 채 깊이 생각하는 모습이 특징이에요.
            삼국시대 사람들은 이 온화한 미소에 평화와 깨달음의 바람을 담았답니다.
          </AssistantMessage>
          <UserMessage>왜 손을 얼굴에 대고 있나요?</UserMessage>
          <AssistantMessage>
            세상의 고민을 깊이 생각하는 모습을 표현한 거예요. 이런 자세를
            ‘반가사유’라고 불러요.
          </AssistantMessage>
          <UserMessage>실제로 보면 얼마나 큰가요?</UserMessage>
          <AssistantMessage>
            높이는 약 93.5cm예요. 가까이에서 보면 섬세한 옷 주름과 잔잔한 미소가
            더 잘 보여요.
          </AssistantMessage>
          {messages.map((message, index) => (
            <div key={`${message}-${index}`} className="new-message-pair">
              <UserMessage>{message}</UserMessage>
              {!thinking && index === messages.length - 1 && (
                <AssistantMessage>
                  좋은 질문이에요. 이 유물은 얇은 금동으로 섬세하게 만들어져,
                  시대를 지나 지금까지 특별한 아름다움을 전하고 있어요.
                </AssistantMessage>
              )}
            </div>
          ))}
          {thinking && <AssistantMessage waiting />}
          <div ref={chatEndRef} />
        </div>

        <div className="composer">
          <div className="suggestions" aria-label="추천 질문">
            {[
              "언제 만들어졌어요?",
              "왜 이런 이름이에요?",
              "어떤 재료로 만들었어요?",
            ].map((question) => (
              <Button
                key={question}
                className="suggestion-chip"
                onClick={() => submitQuestion(question)}
              >
                {question}
              </Button>
            ))}
          </div>
          <form className="input-row" onSubmit={handleSubmit}>
            <input
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="궁금한 점을 물어보세요"
              aria-label="질문 입력"
            />
            <Button
              type="submit"
              className="send-button"
              ariaLabel="질문 보내기"
            >
              <SendIcon />
            </Button>
          </form>
        </div>
      </section>
    </div>
  );
}

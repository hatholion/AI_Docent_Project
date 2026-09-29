import { FormEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Button, Mascot, SendIcon, XIcon } from "./common";
import { ApiError, fetchChatHistory, sendChat } from "../api/docent";
import type { VisitorType } from "../types";

type ChatItem = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  // 답변 실패 시 재시도할 질문
  failedQuestion?: string;
};

const SUGGESTED_QUESTIONS: Record<VisitorType, string[]> = {
  child: ["이건 뭐예요?", "누가 만들었어요?", "왜 이렇게 생겼어요?"],
  general: ["언제 만들어졌어요?", "왜 이런 이름이에요?", "어떤 재료로 만들었어요?"],
  expert: ["제작 기법은 무엇인가요?", "양식상 특징은 무엇인가요?", "비교할 만한 유물이 있나요?"],
};

// 같은 유물·관람객 유형으로 다시 들어오면 이전 대화를 이어간다 (feature_spec 4.3)
const sessionKey = (artifactId: string, visitorType: VisitorType) =>
  `docent-session:${artifactId}:${visitorType}`;

function readStoredSession(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStoredSession(key: string, sessionId: string | null) {
  try {
    if (sessionId) window.sessionStorage.setItem(key, sessionId);
    else window.sessionStorage.removeItem(key);
  } catch {
    // 저장소를 쓸 수 없으면 새 세션으로 동작한다
  }
}

export const isAbort = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";

export const errorMessage = (error: unknown) =>
  error instanceof ApiError ? error.message : "잠시 후 다시 시도해주세요.";

let nextChatId = 0;
const chatId = () => `chat-${nextChatId++}`;

export function AssistantMessage({
  children,
  sources,
  waiting = false,
  error = false,
}: {
  children?: ReactNode;
  sources?: string[];
  waiting?: boolean;
  error?: boolean;
}) {
  return (
    <div className="message-row assistant-row">
      <Mascot small />
      <div className="message-stack">
        <div
          className={`bubble assistant-bubble ${waiting ? "thinking" : ""} ${error ? "error-bubble" : ""}`}
        >
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
        {sources && sources.length > 0 && (
          <span className="source">출처: {sources.join(", ")}</span>
        )}
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

// 닫혀 있어도 마운트를 유지해 대화 내용과 진행 중인 요청을 보존한다
export function ChatPanel({
  artifactId,
  artifactName,
  visitorType,
  open,
  onClose,
}: {
  artifactId: string;
  artifactName: string;
  visitorType: VisitorType;
  open: boolean;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [text, setText] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const chatAbortRef = useRef<AbortController | null>(null);

  const storageKey = sessionKey(artifactId, visitorType);

  useEffect(() => {
    const storedSessionId = readStoredSession(storageKey);
    setMessages([]);
    setSessionId(null);
    if (!storedSessionId) return;

    const controller = new AbortController();
    fetchChatHistory(storedSessionId, controller.signal)
      .then((history) => {
        setSessionId(history.session_id);
        setMessages(
          history.messages.map((message) => ({
            id: chatId(),
            role: message.role,
            content: message.content,
          })),
        );
      })
      .catch((error) => {
        if (isAbort(error)) return;
        // 만료되거나 없는 세션이면 새 대화로 시작
        writeStoredSession(storageKey, null);
      });
    return () => controller.abort();
  }, [storageKey]);

  useEffect(() => () => chatAbortRef.current?.abort(), []);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus({ preventScroll: true });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  useEffect(() => {
    // scrollIntoView는 바깥 화면까지 스크롤하므로 대화 영역만 스크롤한다
    const conversation = conversationRef.current;
    conversation?.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
  }, [messages, pending, open]);

  const ask = async (question: string) => {
    const controller = new AbortController();
    chatAbortRef.current = controller;
    setPending(true);
    try {
      const response = await sendChat(
        {
          artifact_id: artifactId,
          visitor_type: visitorType,
          ...(sessionId ? { session_id: sessionId } : {}),
          question,
        },
        controller.signal,
      );
      setSessionId(response.session_id);
      writeStoredSession(storageKey, response.session_id);
      setMessages((current) => [
        ...current,
        {
          id: chatId(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (error) {
      if (isAbort(error)) return;
      setMessages((current) => [
        ...current,
        {
          id: chatId(),
          role: "assistant",
          content: `답변을 가져오지 못했어요. ${errorMessage(error)}`,
          failedQuestion: question,
        },
      ]);
    } finally {
      if (chatAbortRef.current === controller) {
        chatAbortRef.current = null;
        setPending(false);
      }
    }
  };

  const submitQuestion = (question: string) => {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || pending) return;
    setMessages((current) => [
      ...current,
      { id: chatId(), role: "user", content: cleanQuestion },
    ]);
    setText("");
    void ask(cleanQuestion);
  };

  const retry = (item: ChatItem) => {
    if (!item.failedQuestion || pending) return;
    setMessages((current) => current.filter((message) => message.id !== item.id));
    void ask(item.failedQuestion);
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submitQuestion(text);
  };

  return (
    <section
      className={`chat-panel chat-drawer ${open ? "chat-drawer-open" : ""}`}
      role="dialog"
      aria-label={`${artifactName} 챗봇`}
      aria-hidden={!open}
      inert={!open}
    >
      <div className="chat-heading">
        <div>
          <span className="section-kicker">민속이와 더 알아보기</span>
          <p>{artifactName} 이야기</p>
        </div>
        <Button className="close-button chat-close" onClick={onClose} ariaLabel="챗봇 닫기">
          <XIcon />
        </Button>
      </div>

      <div className="conversation" aria-live="polite" ref={conversationRef}>
        <AssistantMessage>
          {artifactName}에 대해 궁금한 점을 물어보세요! 민속이가 알려드릴게요.
        </AssistantMessage>
        {messages.map((item) =>
          item.role === "user" ? (
            <UserMessage key={item.id}>{item.content}</UserMessage>
          ) : (
            <AssistantMessage
              key={item.id}
              sources={item.sources}
              error={Boolean(item.failedQuestion)}
            >
              {item.content}
              {item.failedQuestion && (
                <Button
                  className="retry-button"
                  onClick={() => retry(item)}
                  disabled={pending}
                >
                  다시 시도
                </Button>
              )}
            </AssistantMessage>
          ),
        )}
        {pending && <AssistantMessage waiting />}
      </div>

      <div className="composer">
        <div className="suggestions" aria-label="추천 질문">
          {SUGGESTED_QUESTIONS[visitorType].map((question) => (
            <Button
              key={question}
              className="suggestion-chip"
              onClick={() => submitQuestion(question)}
              disabled={pending}
            >
              {question}
            </Button>
          ))}
        </div>
        <form className="input-row" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="궁금한 점을 물어보세요"
            aria-label="질문 입력"
            maxLength={500}
          />
          <Button
            type="submit"
            className="send-button"
            ariaLabel="질문 보내기"
            disabled={pending || !text.trim()}
          >
            <SendIcon />
          </Button>
        </form>
      </div>
    </section>
  );
}

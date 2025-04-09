import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './index.css';
import './App.css';
import Header from './components/Header';
import { FiSend } from 'react-icons/fi';

function App() {
  useEffect(() => {
    document.title = "상명대학교 챗봇 SAMI";
  }, []);

  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState('');
  const [loadingDots, setLoadingDots] = useState('');
  const [error, setError] = useState('');
  const chatContainerRef = useRef(null);

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleSubmit = async () => {
    if (!question.trim()) {
      alert("질문을 입력해주세요.");
      return;
    }
  
    if (loading) return;
  
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const newMessage = { role: 'user', text: question, timestamp: now };
    const loadingMessage = {
      role: 'system',
      text: 'SAMI_LOADING',
      timestamp: ''
    };    
  
    setMessages((prev) => [...prev, newMessage, loadingMessage]);
    setLoading(true);
    setError('');
    setQuestion('');
    setTimeout(() => setQuestion(''), 0);
  
    try {
      const response = await axios.post('http://localhost:8000/ask', {
        question: question,
      });
      const answer = response.data.answer;
      const botNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const newAnswer = { role: 'system', text: answer, timestamp: botNow };
  
      // 로딩 메시지를 답변으로 교체
      setMessages((prev) => [...prev.slice(0, -1), newAnswer]);
    } catch (err) {
      setError('서버와의 통신에 문제가 발생했습니다.');
      // 로딩 메시지 삭제
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };  

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !loading) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const formatMessage = (text) => {
    const urlRegex = /(https?:\/\/[^\s)]+)/g;
    return text.split(urlRegex).map((part, index) => {
      if (part.match(/^https?:\/\//)) {
        return (
          <a key={index} href={part} target="_blank" rel="noopener noreferrer" style={{ color: "blue", textDecoration: "underline" }}>
            {part}
          </a>
        );
      }
      return part;
    });
  };

  useEffect(() => {
    if (!loading) {
      setLoadingDots('');
      return;
    }
  
    const interval = setInterval(() => {
      setLoadingDots((prev) => {
        if (prev === '') return '.';
        if (prev === '.') return '..';
        if (prev === '..') return '...';
        return '';
      });
    }, 500); // 점 바뀌는 간격
  
    return () => clearInterval(interval);
  }, [loading]);  

  useEffect(() => { // 오토스크롤 기능
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  return (
    <div className="App">
      <Header />
      
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((message, index) => (
          <div key={index} className={`message-wrapper ${message.role}`}>
          <div className={`message ${message.role}${message.text === 'SAMI_LOADING' ? ' loading' : ''}`}>
            {message.text === 'SAMI_LOADING' ? (
              <span>{`SAMI가 답변 생성 중입니다${loadingDots}`}</span>
            ) : (
              formatMessage(message.text)
            )}
          </div>

          {message.timestamp && (
            <div className="timestamp">{message.timestamp}</div>
          )}
        </div>        
        ))}
      </div>

      <div className="input-wrapper">
        <textarea
          value={question}
          onChange={handleQuestionChange}
          onKeyDown={handleKeyDown}
          placeholder="무엇이든 물어보세요"
          rows="3"
        />
        <button className="send-button" onClick={handleSubmit}>
          <FiSend />
        </button>
      </div>

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}


export default App;

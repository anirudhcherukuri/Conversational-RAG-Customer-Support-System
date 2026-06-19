import React, { useState, useEffect, useRef } from "react";

// Helper to get backend API URL
const getBackendUrl = () => {
  if (window.location.hostname === "localhost") {
    return "http://localhost:8000";
  }
  // Deployed environment - will rely on netlify redirects or custom config
  return "";
};

const BACKEND_URL = getBackendUrl();

export default function App() {
  const [sessions, setSessions] = useState(["session_default"]);
  const [activeSession, setActiveSession] = useState("session_default");
  const [chatHistory, setChatHistory] = useState({
    session_default: [
      {
        role: "assistant",
        content: "Hello! I am your Conversational RAG Support Assistant. You can upload custom files (FAQs, manuals) in the sidebar or ask questions about our pre-loaded policies.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]
  });
  
  const [inputText, setInputText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  
  // RAG Trace details
  const [logs, setLogs] = useState([]);
  const [rawDocuments, setRawDocuments] = useState([]);
  const [rerankedDocuments, setRerankedDocuments] = useState([]);
  const [faithfulnessScore, setFaithfulnessScore] = useState(0.0);
  const [faithfulnessReason, setFaithfulnessReason] = useState("");
  const [attempts, setAttempts] = useState(0);
  const [hasQueried, setHasQueried] = useState(false);

  const messagesListRef = useRef(null);
  const fileInputRef = useRef(null);

  // Prevent browser scroll restoration and input-focus window scroll issues
  useEffect(() => {
    const handleScroll = () => {
      window.scrollTo(0, 0);
      if (document.documentElement) {
        document.documentElement.scrollTop = 0;
      }
      if (document.body) {
        document.body.scrollTop = 0;
      }
    };
    
    window.addEventListener("scroll", handleScroll);
    handleScroll(); // Initial snap
    
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // Auto scroll to bottom of chat container programmatically without window scroll leakage
  useEffect(() => {
    if (messagesListRef.current) {
      messagesListRef.current.scrollTo({
        top: messagesListRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [chatHistory, activeSession]);

  // Fetch documents uploaded for the active session
  const fetchSessionDocuments = async (sid) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/sessions/${sid}/documents`);
      if (res.ok) {
        const data = await res.json();
        setUploadedFiles(data.documents || []);
      }
    } catch (e) {
      console.error("Error fetching session documents:", e);
    }
  };

  useEffect(() => {
    fetchSessionDocuments(activeSession);
    // Initialize trace for session view
    setLogs([`Switched to session: ${activeSession}`]);
    setRawDocuments([]);
    setRerankedDocuments([]);
    setFaithfulnessScore(0.0);
    setFaithfulnessReason("");
    setAttempts(0);
    setHasQueried(false);
  }, [activeSession]);

  const handleCreateSession = () => {
    const newId = `session_${Date.now()}`;
    setSessions(prev => [...prev, newId]);
    setChatHistory(prev => ({
      ...prev,
      [newId]: [
        {
          role: "assistant",
          content: "Welcome to a new support session. Upload custom text/PDF documents to start querying them with hybrid retrieval.",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]
    }));
    setActiveSession(newId);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", activeSession);

    setLogs(prev => [...prev, `Uploading document: ${file.name}...`]);

    try {
      const res = await fetch(`${BACKEND_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setLogs(prev => [...prev, `[SUCCESS] File uploaded. Extracted ${data.chunks_count} chunks.`]);
        fetchSessionDocuments(activeSession);
      } else {
        const err = await res.json();
        setLogs(prev => [...prev, `[ERROR] Upload failed: ${err.detail || 'Unknown error'}`]);
      }
    } catch (e) {
      setLogs(prev => [...prev, `[ERROR] Network error during upload: ${e.message}`]);
    }
    // Reset file input
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClearSessionData = async (sid) => {
    if (!confirm("Are you sure you want to delete all uploaded files and index for this session?")) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/sessions/${sid}`, {
        method: "DELETE"
      });
      if (res.ok) {
        setLogs(prev => [...prev, "Session data cleared successfully."]);
        fetchSessionDocuments(sid);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isGenerating) return;

    const userMessage = {
      role: "user",
      content: inputText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // Update chat history locally
    setChatHistory(prev => ({
      ...prev,
      [activeSession]: [...(prev[activeSession] || []), userMessage]
    }));

    setInputText("");
    setIsGenerating(true);
    
    // Clear previous trace
    setLogs([`Question submitted: "${userMessage.content}"`]);
    setRawDocuments([]);
    setRerankedDocuments([]);
    setFaithfulnessScore(0.0);
    setFaithfulnessReason("");
    setAttempts(0);
    setHasQueried(true);

    // Prepare assistant response slot
    const assistantMessageIndex = (chatHistory[activeSession] || []).length + 1;
    
    // Setup message placeholder in state
    setChatHistory(prev => ({
      ...prev,
      [activeSession]: [
        ...(prev[activeSession] || []),
        {
          role: "assistant",
          content: "Analyzing query and retrieving context...",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isStreaming: true
        }
      ]
    }));

    try {
      // Connect to the streaming endpoint
      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: activeSession,
          question: userMessage.content,
          chat_history: (chatHistory[activeSession] || []).map(m => ({
            role: m.role,
            content: m.content
          })),
          confidence_threshold: parseFloat(confidenceThreshold)
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // save trailing partial line

        for (const line of lines) {
          if (line.trim().startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.node === "retrieve") {
                if (data.raw_documents?.length > 0) {
                  setRawDocuments(data.raw_documents);
                }
                if (data.logs?.length > 0) {
                  setLogs(prev => [...prev, ...data.logs.filter(l => !prev.includes(l))]);
                }
              } 
              
              else if (data.node === "rerank") {
                if (data.reranked_documents?.length > 0) {
                  setRerankedDocuments(data.reranked_documents);
                }
                if (data.logs?.length > 0) {
                  setLogs(prev => [...prev, ...data.logs.filter(l => !prev.includes(l))]);
                }
              } 
              
              else if (data.node === "generate") {
                // LLM output is starting to compile
                if (data.generation) {
                  setChatHistory(prev => {
                    const currentHistory = [...prev[activeSession]];
                    currentHistory[assistantMessageIndex] = {
                      role: "assistant",
                      content: data.generation,
                      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                      isStreaming: true
                    };
                    return { ...prev, [activeSession]: currentHistory };
                  });
                }
                if (data.logs?.length > 0) {
                  setLogs(prev => [...prev, ...data.logs.filter(l => !prev.includes(l))]);
                }
              } 
              
              else if (data.node === "guardrail") {
                setFaithfulnessScore(data.faithfulness_score);
                setFaithfulnessReason(data.faithfulness_reason);
                if (data.logs?.length > 0) {
                  setLogs(prev => [...prev, ...data.logs.filter(l => !prev.includes(l))]);
                }
              } 
              
              else if (data.node === "complete") {
                // Finished generation
                setChatHistory(prev => {
                  const currentHistory = [...prev[activeSession]];
                  currentHistory[assistantMessageIndex] = {
                    role: "assistant",
                    content: data.generation,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    isStreaming: false
                  };
                  return { ...prev, [activeSession]: currentHistory };
                });
                
                setFaithfulnessScore(data.faithfulness_score);
                setFaithfulnessReason(data.faithfulness_reason);
                setRerankedDocuments(data.reranked_documents || []);
                setRawDocuments(data.raw_documents || []);
                if (data.logs?.length > 0) {
                  setLogs(data.logs);
                }
              } 
              
              else if (data.node === "error") {
                setLogs(prev => [...prev, `[ERROR] Stream error: ${data.message}`]);
              }
            } catch (err) {
              console.error("Error parsing streaming SSE line:", err, line);
            }
          }
        }
      }
    } catch (e) {
      console.error("Streaming chat connection failed:", e);
      setLogs(prev => [...prev, `[ERROR] Failed to connect to server: ${e.message}`]);
      setChatHistory(prev => {
        const currentHistory = [...prev[activeSession]];
        currentHistory[assistantMessageIndex] = {
          role: "assistant",
          content: "I'm sorry, I encountered a connection error while trying to generate a response. Please verify the backend is running.",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isError: true
        };
        return { ...prev, [activeSession]: currentHistory };
      });
    } finally {
      setIsGenerating(false);
    }
  };

  // Convert Score to percentage progress offset
  const progressOffset = 314 - (314 * faithfulnessScore);

  // Helper to get color class based on score
  const getScoreColor = (score) => {
    if (score >= confidenceThreshold) return "#00e5ff"; // neon cyan
    if (score >= 0.2) return "#ffcb6b"; // orange/yellow
    return "#f43f5e"; // rose/magenta
  };

  return (
    <div className="app-container">
      {/* Sidebar Panel */}
      <aside className="sidebar glass-panel">
        <div className="brand">
          <div className="brand-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
              <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
          </div>
          <div className="brand-name">Conversational RAG</div>
        </div>

        <div>
          <h3 className="section-title">Support Sessions</h3>
          <button className="btn-new-session" onClick={handleCreateSession} style={{ width: "100%", marginBottom: "12px" }}>
            + New Support Ticket
          </button>
          
          <div className="sessions-container">
            {sessions.map(sid => (
              <div 
                key={sid} 
                className={`session-item ${sid === activeSession ? 'active' : ''}`}
                onClick={() => setActiveSession(sid)}
              >
                <span className="session-name">
                  {sid === "session_default" ? "Primary FAQ Channel" : `Ticket #${sid.split('_')[1]?.slice(-4) || 'Custom'}`}
                </span>
                {sid !== "session_default" && (
                  <span className="file-delete" style={{ display: 'flex', alignItems: 'center' }} onClick={(e) => {
                    e.stopPropagation();
                    // Remove from list
                    setSessions(prev => prev.filter(x => x !== sid));
                    if (activeSession === sid) setActiveSession("session_default");
                    handleClearSessionData(sid);
                  }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="section-title">Knowledge Ingestion</h3>
          <div className="upload-box" onClick={() => fileInputRef.current?.click()}>
            <div className="upload-icon" style={{ display: 'flex', justifyContent: 'center' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'hsl(var(--secondary))', marginBottom: '8px' }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div className="upload-text">Upload Knowledge base</div>
            <div className="upload-subtext">Supports PDF, TXT, MD up to 10MB</div>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: "none" }} 
              accept=".txt,.md,.pdf" 
              onChange={handleFileUpload} 
            />
          </div>

          {uploadedFiles.length > 0 ? (
            <div style={{ marginTop: "14px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "11px", color: "hsl(var(--text-muted))" }}>SESSION INDEX:</span>
                <span className="file-delete" style={{ fontSize: "10px" }} onClick={() => handleClearSessionData(activeSession)}>
                  Clear All
                </span>
              </div>
              <div className="uploaded-files">
                {uploadedFiles.map((file, idx) => (
                  <div key={idx} className="file-pill">
                    <span className="file-name">📄 {file}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: "11px", color: "hsl(var(--text-muted))", marginTop: "10px", textAlign: "center" }}>
              Querying default general FAQ knowledge base.
            </div>
          )}
        </div>

        <div className="settings-section">
          <h3 className="section-title">RAG Controls</h3>
          
          <div className="slider-group">
            <div className="slider-label">
              <span>Guardrail Confidence Threshold</span>
              <span className="slider-value">{confidenceThreshold}</span>
            </div>
            <input 
              type="range" 
              min="0.0" 
              max="1.0" 
              step="0.05" 
              value={confidenceThreshold} 
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))} 
            />
          </div>
        </div>
      </aside>

      {/* Main Chat Panel */}
      <main className="chat-container glass-panel">
        <div className="chat-messages-pane">
          <header className="chat-header">
            <div>
              <h2 className="chat-title">
                {activeSession === "session_default" ? "Standard FAQ Bot" : `Session - Support Ticket #${activeSession.split('_')[1]?.slice(-4)}`}
              </h2>
              <span className="chat-subtitle">Powered by Groq Mixtral & LangGraph</span>
            </div>
            {isGenerating && (
              <div className="pulse-loader">
                <div></div><div></div><div></div>
              </div>
            )}
          </header>

          <div ref={messagesListRef} className="messages-list">
            {(chatHistory[activeSession] || []).map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className="message-bubble">
                  {msg.content}
                </div>
                <div className="message-meta">
                  <span>{msg.role === "user" ? "You" : "Agent"}</span>
                  <span>•</span>
                  <span>{msg.timestamp}</span>
                </div>
              </div>
            ))}
          </div>

          <form className="chat-input-bar" onSubmit={handleSendMessage}>
            <input 
              type="text" 
              className="chat-input"
              value={inputText}
              placeholder="Ask a question or request information..."
              onChange={(e) => setInputText(e.target.value)}
              disabled={isGenerating}
            />
            <button className="btn-send" type="submit" aria-label="Send message" disabled={isGenerating || !inputText.trim()}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </form>
        </div>

        {/* Trace Panel */}
        <section className="rag-trace-pane">
          <div className="trace-section">
            <h3 className="section-title">Guardrail Assessment</h3>
            <div className="gauge-container">
              <div className="radial-gauge">
                <svg width="120" height="120">
                  <circle className="bg-circle" cx="60" cy="60" r="50"></circle>
                  <circle 
                    className="value-circle" 
                    cx="60" 
                    cy="60" 
                    r="50"
                    style={{
                      strokeDashoffset: hasQueried ? progressOffset : 314,
                      stroke: hasQueried ? getScoreColor(faithfulnessScore) : "rgba(255, 255, 255, 0.1)"
                    }}
                  ></circle>
                </svg>
                <div className="gauge-text">
                  <span className="gauge-number" style={{ color: hasQueried ? getScoreColor(faithfulnessScore) : "hsl(var(--text-muted))" }}>
                    {hasQueried ? faithfulnessScore.toFixed(2) : "--"}
                  </span>
                  <span className="gauge-percent">Faithful</span>
                </div>
              </div>
              
              <div className={`guardrail-status-pill ${
                !hasQueried 
                  ? 'status-neutral' 
                  : faithfulnessScore >= confidenceThreshold 
                    ? 'status-passed' 
                    : 'status-failed'
              }`}>
                {!hasQueried 
                  ? "Awaiting Query" 
                  : faithfulnessScore >= confidenceThreshold 
                    ? "Confidence High" 
                    : "Guardrail Flagged"
                }
              </div>
              
              {hasQueried && faithfulnessReason && (
                <div style={{ fontSize: "11px", color: "hsl(var(--text-secondary))", marginTop: "12px", textAlign: "center", fontStyle: "italic" }}>
                  "{faithfulnessReason}"
                </div>
              )}
            </div>
          </div>

          <div className="trace-section">
            <h3 className="section-title">Execution logs</h3>
            <div className="terminal-console">
              {logs.length > 0 ? (
                logs.map((log, idx) => {
                  let type = "info";
                  if (log.includes("[SUCCESS]")) type = "success";
                  if (log.includes("[ERROR]")) type = "error";
                  if (log.includes("[WARNING]") || log.includes("below threshold")) type = "warning";
                  return (
                    <div key={idx} className={`log-entry ${type}`}>
                      &gt; {log}
                    </div>
                  );
                })
              ) : (
                <div style={{ color: "hsl(var(--text-muted))" }}>Waiting for query execution...</div>
              )}
            </div>
          </div>

          <div className="trace-section" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <h3 className="section-title">Context Retrieved</h3>
            <div className="doc-trace-list" style={{ overflowY: "auto", flex: 1 }}>
              {rerankedDocuments.length > 0 ? (
                rerankedDocuments.map((doc, idx) => (
                  <div key={idx} className="doc-trace-item">
                    <div className="doc-trace-header">
                      <span className="doc-trace-source">📄 {doc.metadata?.source || 'Chroma chunk'}</span>
                      <span className="doc-trace-score">Re-Score: {doc.rerank_score?.toFixed(3)}</span>
                    </div>
                    <div className="doc-trace-body">
                      {doc.text}
                    </div>
                  </div>
                ))
              ) : rawDocuments.length > 0 ? (
                rawDocuments.map((doc, idx) => (
                  <div key={idx} className="doc-trace-item">
                    <div className="doc-trace-header">
                      <span className="doc-trace-source">📄 {doc.metadata?.source || 'Chroma chunk'}</span>
                      <span className="doc-trace-score">RRF: {doc.rrf_score?.toFixed(3)}</span>
                    </div>
                    <div className="doc-trace-body">
                      {doc.text}
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: "12px", color: "hsl(var(--text-muted))", textAlign: "center", marginTop: "20px" }}>
                  No docs retrieved yet.
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

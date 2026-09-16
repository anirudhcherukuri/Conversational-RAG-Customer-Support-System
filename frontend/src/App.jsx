import React, { useState, useEffect, useRef } from "react";
import "./App.css";

// ============================================================
// BACKEND API CONFIGURATION
// ============================================================

const getBackendUrl = () => {
  // Local development
  if (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  ) {
    return "http://localhost:8000";
  }

  // Production - call Render backend directly.
  // This avoids depending on Netlify API redirects.
  return "https://conversational-rag-customer-support.onrender.com";
};

const BACKEND_URL = getBackendUrl();

// ============================================================
// RESPONSE HELPERS
// ============================================================

/**
 * Safely read a response body without assuming JSON.
 *
 * This prevents:
 * "Unexpected end of JSON input"
 */
const readResponseData = async (response) => {
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();

  if (!text.trim()) {
    return {
      data: null,
      text: "",
      contentType,
    };
  }

  if (contentType.includes("application/json")) {
    try {
      return {
        data: JSON.parse(text),
        text,
        contentType,
      };
    } catch (error) {
      console.error("Invalid JSON response:", text, error);

      return {
        data: null,
        text,
        contentType,
      };
    }
  }

  return {
    data: null,
    text,
    contentType,
  };
};

/**
 * Convert an unsuccessful HTTP response into a useful error message.
 */
const getHttpErrorMessage = async (response) => {
  const { data, text } = await readResponseData(response);

  if (data?.detail) {
    return data.detail;
  }

  if (data?.message) {
    return data.message;
  }

  if (text?.trim()) {
    return text.trim();
  }

  return `Server returned HTTP ${response.status} ${response.statusText || ""}`.trim();
};

// ============================================================
// APP
// ============================================================

export default function App() {
  // ==========================================================
  // SESSION STATE
  // ==========================================================

  const [sessions, setSessions] = useState(["session_default"]);

  const [activeSession, setActiveSession] =
    useState("session_default");

  const [chatHistory, setChatHistory] = useState({
    session_default: [
      {
        role: "assistant",
        content:
          "Hello! I am your Conversational RAG Support Assistant. You can upload custom files (FAQs, manuals) in the sidebar or ask questions about our pre-loaded policies.",
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ],
  });

  // ==========================================================
  // CHAT STATE
  // ==========================================================

  const [inputText, setInputText] = useState("");

  const [isGenerating, setIsGenerating] =
    useState(false);

  const [confidenceThreshold, setConfidenceThreshold] =
    useState(0.4);

  const [uploadedFiles, setUploadedFiles] =
    useState([]);

  // ==========================================================
  // RAG TRACE STATE
  // ==========================================================

  const [logs, setLogs] = useState([]);

  const [rawDocuments, setRawDocuments] =
    useState([]);

  const [rerankedDocuments, setRerankedDocuments] =
    useState([]);

  const [faithfulnessScore, setFaithfulnessScore] =
    useState(0.0);

  const [faithfulnessReason, setFaithfulnessReason] =
    useState("");

  const [attempts, setAttempts] = useState(0);

  const [hasQueried, setHasQueried] = useState(false);

  // ==========================================================
  // REFS
  // ==========================================================

  const messagesListRef = useRef(null);

  const fileInputRef = useRef(null);

  // ==========================================================
  // PREVENT WINDOW SCROLL
  // ==========================================================

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

    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // ==========================================================
  // AUTO SCROLL CHAT
  // ==========================================================

  useEffect(() => {
    if (messagesListRef.current) {
      messagesListRef.current.scrollTo({
        top: messagesListRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [chatHistory, activeSession]);

  // ==========================================================
  // FETCH SESSION DOCUMENTS
  // ==========================================================

  const fetchSessionDocuments = async (sid) => {
    try {
      const url = `${BACKEND_URL}/api/sessions/${encodeURIComponent(
        sid
      )}/documents`;

      const res = await fetch(url, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });

      if (!res.ok) {
        const message = await getHttpErrorMessage(res);

        console.error(
          `Failed to fetch documents: HTTP ${res.status}`,
          message
        );

        setUploadedFiles([]);

        return;
      }

      const { data, text } = await readResponseData(res);

      if (!data) {
        console.error(
          "Documents endpoint returned an invalid/empty response:",
          text
        );

        setUploadedFiles([]);

        return;
      }

      setUploadedFiles(data.documents || []);
    } catch (error) {
      console.error(
        "Error fetching session documents:",
        error
      );

      setUploadedFiles([]);
    }
  };

  // ==========================================================
  // ACTIVE SESSION CHANGE
  // ==========================================================

  useEffect(() => {
    fetchSessionDocuments(activeSession);

    setLogs([
      `Switched to session: ${activeSession}`,
    ]);

    setRawDocuments([]);

    setRerankedDocuments([]);

    setFaithfulnessScore(0.0);

    setFaithfulnessReason("");

    setAttempts(0);

    setHasQueried(false);
  }, [activeSession]);

  // ==========================================================
  // CREATE NEW SESSION
  // ==========================================================

  const handleCreateSession = () => {
    const newId = `session_${Date.now()}`;

    setSessions((prev) => [
      ...prev,
      newId,
    ]);

    setChatHistory((prev) => ({
      ...prev,
      [newId]: [
        {
          role: "assistant",
          content:
            "Welcome to a new support session. Upload custom text/PDF documents to start querying them with hybrid retrieval.",
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
      ],
    }));

    setActiveSession(newId);
  };

  // ==========================================================
  // FILE UPLOAD
  // ==========================================================

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];

    if (!file) {
      return;
    }

    // --------------------------------------------
    // Basic client-side validation
    // --------------------------------------------

    const allowedExtensions = [
      ".pdf",
      ".txt",
      ".md",
    ];

    const lowerName = file.name.toLowerCase();

    const validExtension = allowedExtensions.some(
      (extension) =>
        lowerName.endsWith(extension)
    );

    if (!validExtension) {
      setLogs((prev) => [
        ...prev,
        "[ERROR] Unsupported file type. Please upload PDF, TXT, or MD files.",
      ]);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      return;
    }

    // 10 MB
    const maxSize = 10 * 1024 * 1024;

    if (file.size > maxSize) {
      setLogs((prev) => [
        ...prev,
        "[ERROR] File is larger than the 10MB limit.",
      ]);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      return;
    }

    // --------------------------------------------
    // Create FormData
    // --------------------------------------------

    const formData = new FormData();

    formData.append("file", file);

    formData.append(
      "session_id",
      activeSession
    );

    setLogs((prev) => [
      ...prev,
      `Uploading document: ${file.name}...`,
    ]);

    try {
      const uploadUrl =
        `${BACKEND_URL}/api/upload`;

      console.log(
        "Uploading file to:",
        uploadUrl
      );

      const res = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      // --------------------------------------------
      // Handle HTTP errors safely
      // --------------------------------------------

      if (!res.ok) {
        const errorMessage =
          await getHttpErrorMessage(res);

        console.error(
          "Upload failed:",
          res.status,
          errorMessage
        );

        setLogs((prev) => [
          ...prev,
          `[ERROR] Upload failed: ${errorMessage}`,
        ]);

        return;
      }

      // --------------------------------------------
      // Parse successful response safely
      // --------------------------------------------

      const { data, text } =
        await readResponseData(res);

      if (!data) {
        console.error(
          "Upload returned an empty or invalid response:",
          text
        );

        setLogs((prev) => [
          ...prev,
          "[ERROR] Upload succeeded at HTTP level, but the server returned an invalid response.",
        ]);

        return;
      }

      // --------------------------------------------
      // Success
      // --------------------------------------------

      setLogs((prev) => [
        ...prev,
        `[SUCCESS] File uploaded. Extracted ${
          data.chunks_count ?? 0
        } chunks.`,
      ]);

      // Refresh documents
      await fetchSessionDocuments(
        activeSession
      );
    } catch (error) {
      console.error(
        "Upload request failed:",
        error
      );

      setLogs((prev) => [
        ...prev,
        `[ERROR] Network error during upload: ${error.message}`,
      ]);
    } finally {
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // ==========================================================
  // CLEAR SESSION DATA
  // ==========================================================

  const handleClearSessionData = async (sid) => {
    const confirmed = window.confirm(
      "Are you sure you want to delete all uploaded files and index for this session?"
    );

    if (!confirmed) {
      return;
    }

    try {
      const url =
        `${BACKEND_URL}/api/sessions/${encodeURIComponent(
          sid
        )}`;

      const res = await fetch(url, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errorMessage =
          await getHttpErrorMessage(res);

        setLogs((prev) => [
          ...prev,
          `[ERROR] Failed to clear session: ${errorMessage}`,
        ]);

        return;
      }

      setLogs((prev) => [
        ...prev,
        "Session data cleared successfully.",
      ]);

      await fetchSessionDocuments(sid);
    } catch (error) {
      console.error(
        "Error clearing session:",
        error
      );

      setLogs((prev) => [
        ...prev,
        `[ERROR] Failed to clear session: ${error.message}`,
      ]);
    }
  };

  // ==========================================================
  // HANDLE CHAT MESSAGE
  // ==========================================================

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (
      !inputText.trim() ||
      isGenerating
    ) {
      return;
    }

    // --------------------------------------------
    // Capture current session and question
    // --------------------------------------------

    const currentSession = activeSession;

    const question = inputText.trim();

    // --------------------------------------------
    // User message
    // --------------------------------------------

    const userMessage = {
      role: "user",
      content: question,
      timestamp: new Date().toLocaleTimeString(
        [],
        {
          hour: "2-digit",
          minute: "2-digit",
        }
      ),
    };

    // --------------------------------------------
    // Get current history BEFORE modifying it
    // --------------------------------------------

    const currentHistory =
      chatHistory[currentSession] || [];

    // The assistant placeholder will be inserted
    // after the user message.
    const assistantMessageIndex =
      currentHistory.length + 1;

    // --------------------------------------------
    // Update user message
    // --------------------------------------------

    setChatHistory((prev) => ({
      ...prev,
      [currentSession]: [
        ...(prev[currentSession] || []),
        userMessage,
      ],
    }));

    setInputText("");

    setIsGenerating(true);

    // --------------------------------------------
    // Clear previous trace
    // --------------------------------------------

    setLogs([
      `Question submitted: "${question}"`,
    ]);

    setRawDocuments([]);

    setRerankedDocuments([]);

    setFaithfulnessScore(0.0);

    setFaithfulnessReason("");

    setAttempts(0);

    setHasQueried(true);

    // --------------------------------------------
    // Assistant placeholder
    // --------------------------------------------

    setChatHistory((prev) => ({
      ...prev,
      [currentSession]: [
        ...(prev[currentSession] || []),
        {
          role: "assistant",
          content:
            "Analyzing query and retrieving context...",
          timestamp: new Date().toLocaleTimeString(
            [],
            {
              hour: "2-digit",
              minute: "2-digit",
            }
          ),
          isStreaming: true,
        },
      ],
    }));

    try {
      // ====================================================
      // CONNECT TO STREAMING ENDPOINT
      // ====================================================

      const streamUrl =
        `${BACKEND_URL}/api/chat/stream`;

      console.log(
        "Connecting to streaming endpoint:",
        streamUrl
      );

      const response = await fetch(
        streamUrl,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
            Accept:
              "text/event-stream",
          },

          body: JSON.stringify({
            session_id:
              currentSession,

            question: question,

            chat_history:
              currentHistory.map((message) => ({
                role: message.role,
                content: message.content,
              })),

            confidence_threshold:
              parseFloat(
                confidenceThreshold
              ),
          }),
        }
      );

      // ====================================================
      // HANDLE HTTP ERROR
      // ====================================================

      if (!response.ok) {
        const errorMessage =
          await getHttpErrorMessage(
            response
          );

        throw new Error(
          `HTTP ${response.status}: ${errorMessage}`
        );
      }

      // ====================================================
      // CHECK STREAM BODY
      // ====================================================

      if (!response.body) {
        throw new Error(
          "Server returned no streaming response body."
        );
      }

      // ====================================================
      // STREAM READER
      // ====================================================

      const reader =
        response.body.getReader();

      const decoder =
        new TextDecoder();

      let buffer = "";

      // ====================================================
      // PROCESS SSE EVENT
      // ====================================================

      const processEvent = (rawEvent) => {
        const eventText =
          rawEvent.trim();

        if (!eventText) {
          return;
        }

        // SSE can technically contain multiple
        // lines. Find the data line.
        const dataLine =
          eventText
            .split("\n")
            .find((line) =>
              line.trim().startsWith("data:")
            );

        if (!dataLine) {
          return;
        }

        const jsonText =
          dataLine
            .trim()
            .slice(5)
            .trim();

        if (!jsonText) {
          return;
        }

        try {
          const data =
            JSON.parse(jsonText);

          // ==================================================
          // RETRIEVE NODE
          // ==================================================

          if (
            data.node === "retrieve"
          ) {
            if (
              Array.isArray(
                data.raw_documents
              ) &&
              data.raw_documents.length > 0
            ) {
              setRawDocuments(
                data.raw_documents
              );
            }

            if (
              Array.isArray(data.logs) &&
              data.logs.length > 0
            ) {
              setLogs((prev) => [
                ...prev,
                ...data.logs.filter(
                  (log) =>
                    !prev.includes(log)
                ),
              ]);
            }
          }

          // ==================================================
          // RERANK NODE
          // ==================================================

          else if (
            data.node === "rerank"
          ) {
            if (
              Array.isArray(
                data.reranked_documents
              ) &&
              data.reranked_documents.length >
                0
            ) {
              setRerankedDocuments(
                data.reranked_documents
              );
            }

            if (
              Array.isArray(data.logs) &&
              data.logs.length > 0
            ) {
              setLogs((prev) => [
                ...prev,
                ...data.logs.filter(
                  (log) =>
                    !prev.includes(log)
                ),
              ]);
            }
          }

          // ==================================================
          // GENERATE NODE
          // ==================================================

          else if (
            data.node === "generate"
          ) {
            if (
              typeof data.generation ===
              "string"
            ) {
              setChatHistory((prev) => {
                const history = [
                  ...(prev[
                    currentSession
                  ] || []),
                ];

                history[
                  assistantMessageIndex
                ] = {
                  role: "assistant",

                  content:
                    data.generation,

                  timestamp:
                    new Date().toLocaleTimeString(
                      [],
                      {
                        hour: "2-digit",
                        minute:
                          "2-digit",
                      }
                    ),

                  isStreaming: true,
                };

                return {
                  ...prev,
                  [currentSession]:
                    history,
                };
              });
            }

            if (
              Array.isArray(data.logs) &&
              data.logs.length > 0
            ) {
              setLogs((prev) => [
                ...prev,
                ...data.logs.filter(
                  (log) =>
                    !prev.includes(log)
                ),
              ]);
            }
          }

          // ==================================================
          // GUARDRAIL NODE
          // ==================================================

          else if (
            data.node === "guardrail"
          ) {
            if (
              typeof data.faithfulness_score ===
              "number"
            ) {
              setFaithfulnessScore(
                data.faithfulness_score
              );
            }

            if (
              typeof data.faithfulness_reason ===
              "string"
            ) {
              setFaithfulnessReason(
                data.faithfulness_reason
              );
            }

            if (
              Array.isArray(data.logs) &&
              data.logs.length > 0
            ) {
              setLogs((prev) => [
                ...prev,
                ...data.logs.filter(
                  (log) =>
                    !prev.includes(log)
                ),
              ]);
            }
          }

          // ==================================================
          // COMPLETE NODE
          // ==================================================

          else if (
            data.node === "complete"
          ) {
            setChatHistory((prev) => {
              const history = [
                ...(prev[
                  currentSession
                ] || []),
              ];

              history[
                assistantMessageIndex
              ] = {
                role: "assistant",

                content:
                  data.generation ||
                  "No response was generated.",

                timestamp:
                  new Date().toLocaleTimeString(
                    [],
                    {
                      hour: "2-digit",
                      minute:
                        "2-digit",
                    }
                  ),

                isStreaming: false,
              };

              return {
                ...prev,
                [currentSession]:
                  history,
              };
            });

            if (
              typeof data.faithfulness_score ===
              "number"
            ) {
              setFaithfulnessScore(
                data.faithfulness_score
              );
            }

            if (
              typeof data.faithfulness_reason ===
              "string"
            ) {
              setFaithfulnessReason(
                data.faithfulness_reason
              );
            }

            setRerankedDocuments(
              data.reranked_documents ||
                []
            );

            setRawDocuments(
              data.raw_documents ||
                []
            );

            if (
              Array.isArray(data.logs) &&
              data.logs.length > 0
            ) {
              setLogs(data.logs);
            }

            if (
              typeof data.attempts ===
              "number"
            ) {
              setAttempts(
                data.attempts
              );
            }
          }

          // ==================================================
          // ERROR NODE
          // ==================================================

          else if (
            data.node === "error"
          ) {
            const errorMessage =
              data.message ||
              "Unknown streaming error.";

            setLogs((prev) => [
              ...prev,
              `[ERROR] Stream error: ${errorMessage}`,
            ]);

            setChatHistory((prev) => {
              const history = [
                ...(prev[
                  currentSession
                ] || []),
              ];

              history[
                assistantMessageIndex
              ] = {
                role: "assistant",

                content:
                  `I'm sorry, an error occurred while processing your request.\n\n${errorMessage}`,

                timestamp:
                  new Date().toLocaleTimeString(
                    [],
                    {
                      hour: "2-digit",
                      minute:
                        "2-digit",
                    }
                  ),

                isError: true,

                isStreaming: false,
              };

              return {
                ...prev,
                [currentSession]:
                  history,
              };
            });
          }
        } catch (error) {
          console.error(
            "Error parsing SSE event:",
            error,
            rawEvent
          );

          setLogs((prev) => [
            ...prev,
            `[WARNING] Received an invalid streaming event from the server.`,
          ]);
        }
      };

      // ====================================================
      // READ STREAM
      // ====================================================

      while (true) {
        const {
          value,
          done,
        } = await reader.read();

        if (done) {
          break;
        }

        if (!value) {
          continue;
        }

        buffer += decoder.decode(
          value,
          {
            stream: true,
          }
        );

        // SSE events are separated by blank lines.
        const events =
          buffer.split(/\r?\n\r?\n/);

        // Keep incomplete event
        buffer =
          events.pop() || "";

        for (const event of events) {
          processEvent(event);
        }
      }

      // ====================================================
      // FLUSH TEXT DECODER
      // ====================================================

      buffer += decoder.decode();

      // ====================================================
      // PROCESS FINAL EVENT
      // ====================================================

      if (buffer.trim()) {
        processEvent(buffer);
      }
    } catch (error) {
      // ======================================================
      // CHAT CONNECTION ERROR
      // ======================================================

      console.error(
        "Streaming chat connection failed:",
        error
      );

      const message =
        error?.message ||
        "Unknown connection error.";

      setLogs((prev) => [
        ...prev,
        `[ERROR] Failed to connect to server: ${message}`,
      ]);

      setChatHistory((prev) => {
        const history = [
          ...(prev[currentSession] ||
            []),
        ];

        history[
          assistantMessageIndex
        ] = {
          role: "assistant",

          content:
            `I'm sorry, I encountered an error while trying to generate a response.\n\n${message}`,

          timestamp:
            new Date().toLocaleTimeString(
              [],
              {
                hour: "2-digit",
                minute: "2-digit",
              }
            ),

          isError: true,

          isStreaming: false,
        };

        return {
          ...prev,
          [currentSession]:
            history,
        };
      });
    } finally {
      setIsGenerating(false);
    }
  };

  // ==========================================================
  // SCORE UI
  // ==========================================================

  const progressOffset =
    314 -
    314 * faithfulnessScore;

  // ==========================================================
  // SCORE COLOR
  // ==========================================================

  const getScoreColor = (score) => {
    if (
      score >= confidenceThreshold
    ) {
      return "#00e5ff";
    }

    if (score >= 0.2) {
      return "#ffcb6b";
    }

    return "#f43f5e";
  };

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app-container">

      {/* ==================================================
          SIDEBAR
      ================================================== */}

      <aside className="sidebar glass-panel">

        {/* BRAND */}

        <div className="brand">

          <div className="brand-icon">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />

              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />

              <line
                x1="12"
                y1="22.08"
                x2="12"
                y2="12"
              />
            </svg>
          </div>

          <div className="brand-name">
            Conversational RAG
          </div>

        </div>

        {/* ==================================================
            SUPPORT SESSIONS
        ================================================== */}

        <div>

          <h3 className="section-title">
            Support Sessions
          </h3>

          <button
            className="btn-new-session"
            onClick={handleCreateSession}
            style={{
              width: "100%",
              marginBottom: "12px",
            }}
          >
            + New Support Ticket
          </button>

          <div className="sessions-container">

            {sessions.map((sid) => (
              <div
                key={sid}
                className={`session-item ${
                  sid === activeSession
                    ? "active"
                    : ""
                }`}
                onClick={() =>
                  setActiveSession(sid)
                }
              >

                <span className="session-name">
                  {sid ===
                  "session_default"
                    ? "Primary FAQ Channel"
                    : `Ticket #${
                        sid
                          .split("_")[1]
                          ?.slice(-4) ||
                        "Custom"
                      }`}
                </span>

                {sid !==
                  "session_default" && (
                  <span
                    className="file-delete"
                    style={{
                      display: "flex",
                      alignItems:
                        "center",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();

                      setSessions(
                        (prev) =>
                          prev.filter(
                            (x) =>
                              x !== sid
                          )
                      );

                      if (
                        activeSession ===
                        sid
                      ) {
                        setActiveSession(
                          "session_default"
                        );
                      }

                      handleClearSessionData(
                        sid
                      );
                    }}
                  >

                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line
                        x1="18"
                        y1="6"
                        x2="6"
                        y2="18"
                      />

                      <line
                        x1="6"
                        y1="6"
                        x2="18"
                        y2="18"
                      />
                    </svg>

                  </span>
                )}

              </div>
            ))}

          </div>

        </div>

        {/* ==================================================
            KNOWLEDGE INGESTION
        ================================================== */}

        <div>

          <h3 className="section-title">
            Knowledge Ingestion
          </h3>

          <div
            className="upload-box"
            onClick={() =>
              fileInputRef.current?.click()
            }
          >

            <div
              className="upload-icon"
              style={{
                display: "flex",
                justifyContent:
                  "center",
              }}
            >

              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  color:
                    "hsl(var(--secondary))",
                  marginBottom: "8px",
                }}
              >

                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />

                <polyline points="17 8 12 3 7 8" />

                <line
                  x1="12"
                  y1="3"
                  x2="12"
                  y2="15"
                />

              </svg>

            </div>

            <div className="upload-text">
              Upload Knowledge base
            </div>

            <div className="upload-subtext">
              Supports PDF, TXT, MD up to 10MB
            </div>

            <input
              type="file"
              ref={fileInputRef}
              style={{
                display: "none",
              }}
              accept=".txt,.md,.pdf"
              onChange={
                handleFileUpload
              }
            />

          </div>

          {/* UPLOADED FILES */}

          {uploadedFiles.length >
          0 ? (
            <div
              style={{
                marginTop: "14px",
              }}
            >

              <div
                style={{
                  display: "flex",
                  justifyContent:
                    "space-between",
                  alignItems:
                    "center",
                }}
              >

                <span
                  style={{
                    fontSize: "11px",
                    color:
                      "hsl(var(--text-muted))",
                  }}
                >
                  SESSION INDEX:
                </span>

                <span
                  className="file-delete"
                  style={{
                    fontSize: "10px",
                  }}
                  onClick={() =>
                    handleClearSessionData(
                      activeSession
                    )
                  }
                >
                  Clear All
                </span>

              </div>

              <div className="uploaded-files">

                {uploadedFiles.map(
                  (file, idx) => (
                    <div
                      key={idx}
                      className="file-pill"
                    >

                      <span className="file-name">
                        📄 {file}
                      </span>

                    </div>
                  )
                )}

              </div>

            </div>
          ) : (
            <div
              style={{
                fontSize: "11px",
                color:
                  "hsl(var(--text-muted))",
                marginTop: "10px",
                textAlign: "center",
              }}
            >
              Querying default general FAQ
              knowledge base.
            </div>
          )}

        </div>

        {/* ==================================================
            RAG CONTROLS
        ================================================== */}

        <div className="settings-section">

          <h3 className="section-title">
            RAG Controls
          </h3>

          <div className="slider-group">

            <div className="slider-label">

              <span>
                Guardrail Confidence Threshold
              </span>

              <span className="slider-value">
                {confidenceThreshold}
              </span>

            </div>

            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={
                confidenceThreshold
              }
              onChange={(e) =>
                setConfidenceThreshold(
                  parseFloat(
                    e.target.value
                  )
                )
              }
            />

          </div>

        </div>

      </aside>

      {/* ====================================================
          MAIN CHAT PANEL
      ==================================================== */}

      <main className="chat-container glass-panel">

        {/* ==================================================
            CHAT PANEL
        ================================================== */}

        <div className="chat-messages-pane">

          {/* HEADER */}

          <header className="chat-header">

            <div>

              <h2 className="chat-title">

                {activeSession ===
                "session_default"
                  ? "Standard FAQ Bot"
                  : `Session - Support Ticket #${
                      activeSession
                        .split("_")[1]
                        ?.slice(-4)
                    }`}

              </h2>

              <span className="chat-subtitle">
                Powered by Groq & LangGraph
              </span>

            </div>

            {isGenerating && (
              <div className="pulse-loader">

                <div></div>

                <div></div>

                <div></div>

              </div>
            )}

          </header>

          {/* ==================================================
              MESSAGES
          ================================================== */}

          <div
            ref={messagesListRef}
            className="messages-list"
          >

            {(chatHistory[
              activeSession
            ] || []).map(
              (msg, idx) => (
                <div
                  key={idx}
                  className={`message-wrapper ${
                    msg.role
                  }`}
                >

                  <div
                    className={`message-bubble ${
                      msg.isError
                        ? "error-message"
                        : ""
                    }`}
                  >
                    {msg.content}
                  </div>

                  <div className="message-meta">

                    <span>
                      {msg.role ===
                      "user"
                        ? "You"
                        : "Agent"}
                    </span>

                    <span>•</span>

                    <span>
                      {msg.timestamp}
                    </span>

                  </div>

                </div>
              )
            )}

          </div>

          {/* ==================================================
              CHAT INPUT
          ================================================== */}

          <form
            className="chat-input-bar"
            onSubmit={
              handleSendMessage
            }
          >

            <input
              type="text"
              className="chat-input"
              value={inputText}
              placeholder="Ask a question or request information..."
              onChange={(e) =>
                setInputText(
                  e.target.value
                )
              }
              disabled={isGenerating}
            />

            <button
              className="btn-send"
              type="submit"
              aria-label="Send message"
              disabled={
                isGenerating ||
                !inputText.trim()
              }
            >

              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >

                <line
                  x1="22"
                  y1="2"
                  x2="11"
                  y2="13"
                />

                <polygon points="22 2 15 22 11 13 2 9 22 2" />

              </svg>

            </button>

          </form>

        </div>

        {/* ====================================================
            RAG TRACE PANEL
        ==================================================== */}

        <section className="rag-trace-pane">

          {/* ==================================================
              GUARDRAIL
          ================================================== */}

          <div className="trace-section">

            <h3 className="section-title">
              Guardrail Assessment
            </h3>

            <div className="gauge-container">

              <div className="radial-gauge">

                <svg
                  width="120"
                  height="120"
                >

                  <circle
                    className="bg-circle"
                    cx="60"
                    cy="60"
                    r="50"
                  ></circle>

                  <circle
                    className="value-circle"
                    cx="60"
                    cy="60"
                    r="50"
                    style={{
                      strokeDashoffset:
                        hasQueried
                          ? progressOffset
                          : 314,

                      stroke:
                        hasQueried
                          ? getScoreColor(
                              faithfulnessScore
                            )
                          : "rgba(255, 255, 255, 0.1)",
                    }}
                  ></circle>

                </svg>

                <div className="gauge-text">

                  <span
                    className="gauge-number"
                    style={{
                      color:
                        hasQueried
                          ? getScoreColor(
                              faithfulnessScore
                            )
                          : "hsl(var(--text-muted))",
                    }}
                  >
                    {hasQueried
                      ? faithfulnessScore.toFixed(
                          2
                        )
                      : "--"}
                  </span>

                  <span className="gauge-percent">
                    Faithful
                  </span>

                </div>

              </div>

              {/* STATUS */}

              <div
                className={`guardrail-status-pill ${
                  !hasQueried
                    ? "status-neutral"
                    : faithfulnessScore >=
                      confidenceThreshold
                    ? "status-passed"
                    : "status-failed"
                }`}
              >

                {!hasQueried
                  ? "Awaiting Query"
                  : faithfulnessScore >=
                    confidenceThreshold
                  ? "Confidence High"
                  : "Guardrail Flagged"}

              </div>

              {/* REASON */}

              {hasQueried &&
                faithfulnessReason && (
                  <div
                    style={{
                      fontSize: "11px",
                      color:
                        "hsl(var(--text-secondary))",
                      marginTop: "12px",
                      textAlign:
                        "center",
                      fontStyle:
                        "italic",
                    }}
                  >
                    "{faithfulnessReason}"
                  </div>
                )}

            </div>

          </div>

          {/* ==================================================
              EXECUTION LOGS
          ================================================== */}

          <div className="trace-section">

            <h3 className="section-title">
              Execution logs
            </h3>

            <div className="terminal-console">

              {logs.length > 0 ? (
                logs.map(
                  (log, idx) => {
                    let type = "info";

                    if (
                      log.includes(
                        "[SUCCESS]"
                      )
                    ) {
                      type = "success";
                    }

                    if (
                      log.includes(
                        "[ERROR]"
                      )
                    ) {
                      type = "error";
                    }

                    if (
                      log.includes(
                        "[WARNING]"
                      ) ||
                      log.includes(
                        "below threshold"
                      )
                    ) {
                      type = "warning";
                    }

                    return (
                      <div
                        key={idx}
                        className={`log-entry ${type}`}
                      >
                        &gt; {log}
                      </div>
                    );
                  }
                )
              ) : (
                <div
                  style={{
                    color:
                      "hsl(var(--text-muted))",
                  }}
                >
                  Waiting for query
                  execution...
                </div>
              )}

            </div>

          </div>

          {/* ==================================================
              CONTEXT RETRIEVED
          ================================================== */}

          <div
            className="trace-section"
            style={{
              flex: 1,
              display: "flex",
              flexDirection:
                "column",
            }}
          >

            <h3 className="section-title">
              Context Retrieved
            </h3>

            <div
              className="doc-trace-list"
              style={{
                overflowY:
                  "auto",
                flex: 1,
              }}
            >

              {/* RERANKED DOCUMENTS */}

              {rerankedDocuments.length >
              0 ? (
                rerankedDocuments.map(
                  (doc, idx) => (
                    <div
                      key={idx}
                      className="doc-trace-item"
                    >

                      <div className="doc-trace-header">

                        <span className="doc-trace-source">
                          📄{" "}
                          {doc.metadata
                            ?.source ||
                            "Chroma chunk"}
                        </span>

                        <span className="doc-trace-score">
                          Re-Score:{" "}
                          {typeof doc.rerank_score ===
                          "number"
                            ? doc.rerank_score.toFixed(
                                3
                              )
                            : "N/A"}
                        </span>

                      </div>

                      <div className="doc-trace-body">
                        {doc.text}
                      </div>

                    </div>
                  )
                )

              ) : rawDocuments.length >
                0 ? (

                /* RAW DOCUMENTS */

                rawDocuments.map(
                  (doc, idx) => (
                    <div
                      key={idx}
                      className="doc-trace-item"
                    >

                      <div className="doc-trace-header">

                        <span className="doc-trace-source">
                          📄{" "}
                          {doc.metadata
                            ?.source ||
                            "Chroma chunk"}
                        </span>

                        <span className="doc-trace-score">
                          RRF:{" "}
                          {typeof doc.rrf_score ===
                          "number"
                            ? doc.rrf_score.toFixed(
                                3
                              )
                            : "N/A"}
                        </span>

                      </div>

                      <div className="doc-trace-body">
                        {doc.text}
                      </div>

                    </div>
                  )
                )

              ) : (

                <div
                  style={{
                    fontSize: "12px",
                    color:
                      "hsl(var(--text-muted))",
                    textAlign:
                      "center",
                    marginTop:
                      "20px",
                  }}
                >
                  No docs retrieved
                  yet.
                </div>

              )}

            </div>

          </div>

        </section>

      </main>

    </div>
  );
}

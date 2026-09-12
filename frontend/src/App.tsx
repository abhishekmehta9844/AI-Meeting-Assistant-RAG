import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Square, MessageSquare, FileText, BarChart2, Trash2, CheckCircle, ListTodo, Download, Bot, LogOut, User as UserIcon } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

axios.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

interface Meeting {
  id: number;
  meeting_id: string;
  title: string;
  url: string;
  started_at: string;
  status: string;
}

function AuthScreen({ onLogin }: { onLogin: () => void }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (isLogin) {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);
        const res = await axios.post('http://localhost:8000/api/auth/login', formData, {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        localStorage.setItem('token', res.data.access_token);
        onLogin();
      } else {
        const res = await axios.post('http://localhost:8000/api/auth/register', { email, password });
        localStorage.setItem('token', res.data.access_token);
        onLogin();
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f172a] flex items-center justify-center p-4">
      <div className="bg-slate-800 border border-slate-700 p-8 rounded-2xl shadow-2xl w-full max-w-md">
        <div className="flex justify-center mb-8">
          <div className="bg-blue-600 p-3 rounded-xl shadow-lg shadow-blue-500/20">
            <Bot size={32} className="text-white" />
          </div>
        </div>
        <h2 className="text-2xl font-bold text-white text-center mb-6">
          {isLogin ? 'Welcome back to Nexus' : 'Create an account'}
        </h2>
        
        {error && <div className="bg-red-500/10 text-red-400 border border-red-500/20 p-3 rounded-lg mb-6 text-sm">{error}</div>}
        
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Email</label>
            <input 
              type="email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              required 
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Password</label>
            <input 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
              required 
            />
          </div>
          <button type="submit" disabled={loading} className="w-full bg-blue-600 hover:bg-blue-500 text-white py-3 rounded-xl font-medium mt-2 transition-colors disabled:opacity-50">
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>
        
        <div className="mt-6 text-center text-slate-400 text-sm">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button onClick={() => setIsLogin(!isLogin)} className="text-blue-400 hover:text-blue-300 font-medium">
            {isLogin ? 'Sign up' : 'Log in'}
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [url, setUrl] = useState('');
  const [transcriber, setTranscriber] = useState('sarvam');
  const [loading, setLoading] = useState(false);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);

  useEffect(() => {
    if (isAuthenticated) {
      fetchMeetings();
    }
  }, [isAuthenticated]);

  const fetchMeetings = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/meetings');
      setMeetings(res.data);
    } catch (err: any) {
      console.error(err);
      if (err.response?.status === 401) {
        handleLogout();
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setSelectedMeeting(null);
    setMeetings([]);
  };

  const handleStartBot = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/api/bot/start', {
        url,
        title: "New Meeting",
        transcriber: transcriber
      });
      setUrl('');
      fetchMeetings();
    } catch (err: any) {
      console.error(err);
      if (err.response?.status === 401) handleLogout();
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) {
    return <AuthScreen onLogin={() => setIsAuthenticated(true)} />;
  }

  if (selectedMeeting) {
    return <MeetingWorkspace meeting={selectedMeeting} onBack={() => setSelectedMeeting(null)} />;
  }

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200">
      <nav className="bg-slate-900 border-b border-slate-800 px-8 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-500/20">
            <Bot size={24} className="text-white" />
          </div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Nexus Meetings</h1>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSelectedMeeting({
              id: -1,
              meeting_id: "global",
              title: "Global Super-Chat",
              url: "",
              started_at: new Date().toISOString(),
              status: "completed"
            })}
            className="bg-indigo-600/10 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-600/20 px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-all"
          >
            <MessageSquare size={16} /> Global Search
          </button>
          <div className="h-6 w-px bg-slate-700"></div>
          <button onClick={handleLogout} className="text-slate-400 hover:text-red-400 transition-colors" title="Log out">
            <LogOut size={20} />
          </button>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto p-8 mt-4">
        <div className="bg-slate-800/50 border border-slate-700/50 p-8 rounded-2xl mb-12 shadow-xl backdrop-blur-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none"></div>
          
          <h2 className="text-2xl font-semibold mb-2 text-white">Record a New Meeting</h2>
          <p className="text-slate-400 mb-6">Drop a Google Meet link below and the bot will join instantly to transcribe and index the conversation.</p>
          
          <form onSubmit={handleStartBot} className="flex gap-4 relative z-10">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://meet.google.com/xyz-abcd-efg"
              className="flex-1 bg-slate-900 border border-slate-700 text-white px-5 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder:text-slate-600 shadow-inner"
              required
            />
            <select
              value={transcriber}
              onChange={(e) => setTranscriber(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 shadow-inner"
            >
              <option value="sarvam">Sarvam AI</option>
              <option value="whisper">Whisper</option>
            </select>
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-xl font-medium flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/25"
            >
              {loading ? 'Starting...' : <><Play size={18} /> Join Meeting</>}
            </button>
          </form>
        </div>

        <div>
          <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-2">
            <ListTodo className="text-blue-500" /> Your Meetings
          </h2>
          {meetings.length === 0 ? (
            <div className="text-center py-12 bg-slate-800/30 rounded-2xl border border-slate-700/50 border-dashed">
              <p className="text-slate-400 mb-2">No meetings found.</p>
              <p className="text-sm text-slate-500">Start your first recording above!</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {meetings.map((m) => (
                <div 
                  key={m.meeting_id} 
                  className="bg-slate-800 border border-slate-700 p-6 rounded-2xl flex justify-between items-center hover:border-slate-600 transition-colors group cursor-pointer"
                  onClick={() => setSelectedMeeting(m)}
                >
                  <div>
                    <h3 className="text-lg font-medium text-white mb-1 group-hover:text-blue-400 transition-colors">{m.title}</h3>
                    <div className="flex items-center gap-4 text-sm text-slate-400">
                      <span className="flex items-center gap-1.5">
                        <FileText size={14} /> {new Date(m.started_at).toLocaleString()}
                      </span>
                      <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${m.status === 'completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}`}>
                        {m.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if(window.confirm('Delete this meeting?')) {
                          axios.delete(`http://localhost:8000/api/meetings/${m.meeting_id}`).then(() => fetchMeetings());
                        }
                      }}
                      className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MeetingWorkspace({ meeting, onBack }: { meeting: Meeting, onBack: () => void }) {
  const [activeTab, setActiveTab] = useState('chat');
  
  const handleStopBot = async () => {
    if(window.confirm('Stop the bot and start processing?')) {
      try {
        await axios.post(`http://localhost:8000/api/bot/${meeting.meeting_id}/stop`);
        alert('Bot stop requested! Transcription will now begin.');
      } catch(err) {
        console.error(err);
      }
    }
  };

  const isGlobal = meeting.meeting_id === 'global';
  const tabs = isGlobal ? [
    { id: 'chat', label: 'Global Search', icon: MessageSquare }
  ] : [
    { id: 'chat', label: 'Chat & Search', icon: MessageSquare },
    { id: 'summary', label: 'Meeting Summary', icon: CheckCircle },
    { id: 'analytics', label: 'Analytics', icon: BarChart2 },
    { id: 'transcript', label: 'Raw Transcript', icon: FileText },
  ];

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 flex flex-col">
      <nav className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
            ← Back
          </button>
          <div className="h-6 w-px bg-slate-700"></div>
          <h1 className="font-semibold text-white">{meeting.title}</h1>
          {!isGlobal && <span className="text-sm text-slate-500">ID: {meeting.meeting_id}</span>}
        </div>
        {!isGlobal && meeting.status === 'recording' && (
          <button onClick={handleStopBot} className="bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20 px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors">
            <Square size={16} /> End Meeting Now
          </button>
        )}
      </nav>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-64 bg-slate-900/50 border-r border-slate-800 p-4 shrink-0 flex flex-col gap-2">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`w-full text-left p-3 rounded-xl flex items-center gap-3 transition-all font-medium ${activeTab === t.id ? 'bg-blue-600 shadow-md text-white' : 'hover:bg-slate-800 text-slate-400 hover:text-slate-200'}`}
            >
              <t.icon size={18} /> {t.label}
            </button>
          ))}
        </div>
        <div className="flex-1 bg-[#0f172a] overflow-hidden">
          {activeTab === 'chat' && <ChatTab meeting={meeting} />}
          {activeTab === 'summary' && !isGlobal && <SummaryTab meeting={meeting} />}
          {activeTab === 'transcript' && !isGlobal && <TranscriptTab meeting={meeting} />}
          {activeTab === 'analytics' && !isGlobal && <AnalyticsTab meeting={meeting} />}
        </div>
      </div>
    </div>
  );
}

function ChatTab({ meeting }: { meeting: Meeting }) {
  const [messages, setMessages] = useState<{role:string, content:string}[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': token ? `Bearer ${token}` : ''
        },
        body: JSON.stringify({ meeting_id: meeting.meeting_id, query: userMsg })
      });

      if (!response.body) throw new Error('No body');
      
      setMessages(prev => [...prev, { role: 'bot', content: '' }]);
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.type === 'token') {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const lastIdx = newMsgs.length - 1;
                  newMsgs[lastIdx] = { ...newMsgs[lastIdx], content: newMsgs[lastIdx].content + data.content };
                  return newMsgs;
                });
              }
            } catch (e) {
              // ignore parse error
            }
          }
        }
      }
    } catch(err) {
      console.error(err);
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].content = 'Error getting response.';
        return newMsgs;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto p-8 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500">
            <Bot size={48} className="mb-4 opacity-50" />
            <p className="text-lg">Ask me anything about {meeting.meeting_id === 'global' ? 'all your meetings' : 'this meeting'}!</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl p-4 ${m.role === 'user' ? 'bg-blue-600 text-white rounded-br-sm shadow-md shadow-blue-500/10' : 'bg-slate-700/50 text-slate-200 border border-slate-600/50 rounded-bl-sm shadow-sm'}`}>
              <div className="prose prose-invert max-w-none" dangerouslySetInnerHTML={{ __html: m.content.replace(/\n/g, '<br/>') }} />
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-700/50 border border-slate-600/50 p-4 rounded-2xl rounded-bl-sm animate-pulse flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="p-4 bg-slate-900 border-t border-slate-800">
        <form onSubmit={handleSend} className="flex gap-2 max-w-4xl mx-auto relative">
          <input 
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your question..."
            className="flex-1 bg-slate-800 border border-slate-700 text-white px-5 py-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 shadow-inner pr-12"
          />
          <button 
            type="submit" 
            disabled={loading || !input.trim()}
            className="absolute right-2 top-2 bottom-2 bg-blue-600 hover:bg-blue-500 text-white p-2 w-10 flex items-center justify-center rounded-lg transition-colors disabled:opacity-50"
          >
            ↑
          </button>
        </form>
      </div>
    </div>
  );
}

function SummaryTab({ meeting }: { meeting: Meeting }) {
  const [data, setData] = useState<any>(null);
  
  useEffect(() => {
    axios.get(`http://localhost:8000/api/meetings/${meeting.meeting_id}/summary`)
      .then(res => setData(res.data))
      .catch(err => console.error(err));
  }, [meeting]);

  const handleExport = () => {
    if (!data) return;
    const content = `# Summary\n${data.summary}\n\n# Action Items\n${data.action_items?.map((item: string) => `- ${item}`).join('\n')}\n\n# Key Decisions\n${data.key_decisions?.map((item: string) => `- ${item}`).join('\n')}`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Meeting_Summary_${meeting.meeting_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!data) return <div className="p-8 text-slate-400">Loading...</div>;

  return (
    <div className="p-8 h-full overflow-y-auto max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold text-white">Meeting Summary</h2>
        <button onClick={handleExport} className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-4 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors">
          <Download size={16} /> Export Markdown
        </button>
      </div>
      
      <div className="space-y-8">
        <div className="bg-slate-800/50 border border-slate-700/50 p-6 rounded-2xl">
          <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2"><FileText size={18} className="text-blue-400" /> Overview</h3>
          <p className="text-slate-300 leading-relaxed">{data.summary || "No summary yet."}</p>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-slate-800/50 border border-slate-700/50 p-6 rounded-2xl">
            <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2"><ListTodo size={18} className="text-emerald-400" /> Action Items</h3>
            <ul className="space-y-3">
              {data.action_items?.map((item: string, i: number) => (
                <li key={i} className="flex items-start gap-3 text-slate-300">
                  <span className="mt-1 w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0"></span>
                  <span>{item}</span>
                </li>
              ))}
              {(!data.action_items || data.action_items.length === 0) && <li className="text-slate-500">No action items recorded.</li>}
            </ul>
          </div>
          <div className="bg-slate-800/50 border border-slate-700/50 p-6 rounded-2xl">
            <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2"><CheckCircle size={18} className="text-amber-400" /> Key Decisions</h3>
            <ul className="space-y-3">
              {data.key_decisions?.map((item: string, i: number) => (
                <li key={i} className="flex items-start gap-3 text-slate-300">
                  <span className="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"></span>
                  <span>{item}</span>
                </li>
              ))}
              {(!data.key_decisions || data.key_decisions.length === 0) && <li className="text-slate-500">No key decisions recorded.</li>}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function TranscriptTab({ meeting }: { meeting: Meeting }) {
  const [text, setText] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [oldSpk, setOldSpk] = useState('');
  const [newSpk, setNewSpk] = useState('');

  useEffect(() => {
    fetchTranscript();
  }, [meeting]);

  const fetchTranscript = () => {
    axios.get(`http://localhost:8000/api/meetings/${meeting.meeting_id}/transcript`)
      .then(res => setText(res.data.content))
      .catch(_ => setText('Failed to load transcript.'));
  };

  const handleExport = () => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Transcript_${meeting.meeting_id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleRenameSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!oldSpk || !newSpk) return;
    try {
      const updates = { [oldSpk]: newSpk };
      await axios.patch(`http://localhost:8000/api/meetings/${meeting.meeting_id}/speakers`, updates);
      setIsModalOpen(false);
      setOldSpk('');
      setNewSpk('');
      fetchTranscript(); // reload
    } catch (err) {
      alert('Error updating speakers');
      console.error(err);
    }
  };

  return (
    <div className="p-8 h-full flex flex-col max-w-5xl mx-auto relative">
      <div className="flex justify-between items-center mb-6 shrink-0">
        <h2 className="text-2xl font-bold text-white">Raw Transcript</h2>
        <div className="flex gap-3">
          <button onClick={() => setIsModalOpen(true)} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors">
            <UserIcon size={16} /> Rename Speaker
          </button>
          <button onClick={handleExport} className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-4 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors">
            <Download size={16} /> Export TXT
          </button>
        </div>
      </div>
      <div className="flex-1 bg-slate-900 border border-slate-800 p-6 rounded-2xl overflow-y-auto font-mono text-sm leading-relaxed text-slate-300 whitespace-pre-wrap shadow-inner">
        {text || "Transcript empty or not ready yet."}
      </div>

      {isModalOpen && (
        <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl shadow-2xl w-full max-w-md">
            <h3 className="text-xl font-bold text-white mb-4">Rename Speaker</h3>
            <form onSubmit={handleRenameSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Old Name (e.g. SPEAKER_00 or Unknown)</label>
                <input 
                  value={oldSpk}
                  onChange={e => setOldSpk(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">New Name</label>
                <input 
                  value={newSpk}
                  onChange={e => setNewSpk(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div className="flex justify-end gap-3 mt-2">
                <button type="button" onClick={() => setIsModalOpen(false)} className="px-4 py-2 text-slate-400 hover:text-white transition-colors">Cancel</button>
                <button type="submit" className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg transition-colors">Save</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function AnalyticsTab({ meeting }: { meeting: Meeting }) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    axios.get(`http://localhost:8000/api/meetings/${meeting.meeting_id}/analytics`)
      .then(res => setData(res.data))
      .catch(err => console.error(err));
  }, [meeting]);

  if (!data) return <div className="p-8 text-slate-400">Loading...</div>;
  if (!data.speakerData || data.speakerData.length === 0) return <div className="p-8 text-slate-400">No talk time data available yet.</div>;

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#8b5cf6', '#0ea5e9'];

  return (
    <div className="p-8 h-full overflow-y-auto max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold text-white mb-8">Speaker Analytics</h2>
      
      <div className="grid md:grid-cols-2 gap-8">
        <div className="bg-slate-800/50 border border-slate-700/50 p-6 rounded-2xl flex flex-col">
          <h3 className="text-lg font-medium text-white mb-6 text-center">Share of Voice (Words)</h3>
          <div className="flex-1 min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.speakerData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${((percent || 0) * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {data.speakerData.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem', color: '#f1f5f9' }} />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-800/50 border border-slate-700/50 p-6 rounded-2xl">
          <h3 className="text-lg font-medium text-white mb-6 flex items-center justify-between">
            <span>Talk Time Summary</span>
            <span className="text-sm bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full">Total: {data.totalWords} words</span>
          </h3>
          <div className="space-y-4">
            {data.speakerData.sort((a:any, b:any) => b.value - a.value).map((spk: any, i: number) => (
              <div key={i} className="flex items-center gap-4">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }}></div>
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-200 font-medium">{spk.name}</span>
                    <span className="text-slate-400">{spk.value} words ({spk.percentage}%)</span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${spk.percentage}%`, backgroundColor: COLORS[i % COLORS.length] }}></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;